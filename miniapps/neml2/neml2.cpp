// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
//
//     --------------------------------------------------------------------
//      Solid mechanics problem using NEML2 to handle constitutive updates
//     --------------------------------------------------------------------
//
// Compile with: make neml2
//
// Sample runs:  neml2
//               neml2 -d cpu
//               neml2 -d cuda
//               mpirun -np 4 neml2
//               neml2 -i j2_aoti/model -nt 5
//               neml2 -i chaboche_aoti/model -nt 5 -dt 0.01
//
// Description:  This example code demonstrates the use of MFEM to solve the
//               balance of linear momentum equation in 3D under small
//               deformation assumptions with the constitutive model provided by
//               NEML2. The load is applied over several quasi-static steps via a
//               uniformly ramped displacement boundary condition; history-
//               dependent (plastic) models are supported by carrying per-
//               quadrature-point internal state across steps at every multigrid
//               level (see MaterialStateManager).

#include "operators.hpp"

#include <ATen/Parallel.h> // at::set_num_threads / at::set_num_interop_threads
#include <petscsnes.h>     // SNESGet{Iteration,LinearSolve}Iterations (profiling)

#include "neml2/csrc/dispatchers/SimpleScheduler.h"
#include "neml2/csrc/dispatchers/factory.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;
using namespace mfem::future;

// Solver/preconditioner configuration for the GPU-plasticity study. The fine
// operator is always matrix-free; these knobs vary the outer Krylov method, the
// matrix-free fine smoother, and the assembled-coarse AMG. Defaults reproduce the
// original setup. See miniapps/neml2/benchmarks.md for the combinations.
struct SolverConfig
{
   std::string ksp = "cg";               // cg | gmres | fgmres (outer Krylov)
   std::string smoother = "chebyshev";   // chebyshev | jacobi (matrix-free fine)
   int smoother_order = 2;               // Chebyshev degree
   std::string coarse = "boomeramg";     // boomeramg | gamg | none
   std::string coarse_mode = "inner-cg"; // inner-cg | vcycle
   bool tuned_amg = false;               // plasticity-tuned scalar BoomerAMG params
   bool near_null_space = true;          // rigid-body near-null-space for GAMG
};

// Attach the 6 rigid-body modes (3 translations + 3 rotations) as the near-null
// space of the assembled coarse operator, so PETSc GAMG's smoothed aggregation
// builds good elasticity coarse spaces. Ordering-safe (works for byNODES): each
// mode is projected as a coarse ParGridFunction from a VectorFunctionCoefficient
// of the nodal coordinates, restricted to the true dofs, zeroed on essential
// dofs, then orthonormalized. The MatNullSpace references the vectors and the Mat
// references the null space, so both outlive the local handles here.
static void AttachRigidBodyNullSpace(ParFiniteElementSpace &fes,
                                     const Array<int> &ess_tdofs,
                                     PetscParMatrix &A)
{
   constexpr int nrbm = 6;
   const MPI_Comm comm = fes.GetComm();

   std::vector<Vector> modes(nrbm);
   ParGridFunction gf(&fes);
   for (int m = 0; m < nrbm; ++m)
   {
      VectorFunctionCoefficient c(3, [m](const Vector &x, Vector &v)
      {
         v = 0.0;
         switch (m)
         {
            case 0: v(0) = 1.0; break;                 // translation x
            case 1: v(1) = 1.0; break;                 // translation y
            case 2: v(2) = 1.0; break;                 // translation z
            case 3: v(1) = -x(2); v(2) = x(1); break;  // rotation about x
            case 4: v(0) = x(2); v(2) = -x(0); break;  // rotation about y
            case 5: v(0) = -x(1); v(1) = x(0); break;  // rotation about z
         }
      });
      gf.ProjectCoefficient(c);
      modes[m].SetSize(fes.GetTrueVSize());
      gf.GetTrueDofs(modes[m]);
      modes[m].SetSubVector(ess_tdofs, 0.0);
   }

   // Modified Gram-Schmidt (MatNullSpaceCreate assumes orthonormal vectors).
   for (int m = 0; m < nrbm; ++m)
   {
      for (int k = 0; k < m; ++k)
      {
         modes[m].Add(-InnerProduct(comm, modes[m], modes[k]), modes[k]);
      }
      const real_t nrm = std::sqrt(InnerProduct(comm, modes[m], modes[m]));
      if (nrm > 1e-14) { modes[m] *= 1.0 / nrm; }
   }

   std::vector<std::unique_ptr<PetscParVector>> holders;
   Array<Vec> vecs(nrbm);
   for (int m = 0; m < nrbm; ++m)
   {
      holders.push_back(std::make_unique<PetscParVector>(comm, modes[m], true));
      vecs[m] = *holders.back();
   }
   MatNullSpace sp;
   PetscCallAbort(comm,
                  MatNullSpaceCreate(comm, PETSC_FALSE, nrbm, vecs.GetData(), &sp));
   PetscCallAbort(comm, MatSetNearNullSpace(A, sp));
   PetscCallAbort(comm, MatNullSpaceDestroy(&sp));
}

/// @brief Restrict a *solution* (a linearization point) from one level to the next
/// coarser one, as the column-sum-normalized transpose of the unconstrained
/// prolongation.
///
/// This is deliberately not `GeometricMultigrid::prolongations[level]`'s
/// `MultTranspose`, which is the restriction for residuals. Two separate things
/// make that one wrong for a solution:
///
/// - It sums where it should average. The column sums of a nodal interpolation
///   run from 1 at a corner to 8 in the interior of a 3D patch, so `P^T` inflates
///   the interior of a displacement field by up to 8x.
/// - `GeometricMultigrid` wraps every prolongation in a
///   RectangularConstrainedOperator, which zeroes the essential true dofs on both
///   sides. That is right for a correction, which vanishes there, but it discards
///   the prescribed boundary displacement outright: at the first Newton iterate,
///   where `u` is nonzero only on the loaded face, `P^T u` is identically zero.
///
/// Together they hand the coarse level a state the fine level never visits -- an
/// interior stretched several-fold against a boundary pinned at zero. A
/// preconditioner does not have to be consistent to be valid, but this is far
/// enough outside the fine level's trajectory to drive the local return map past
/// its radius of convergence: under an n=8 power-law slip rule the constitutive
/// residual grows as the 8th power of the overshoot.
///
/// Normalizing the *unconstrained* transpose fixes both. Every coarse dof
/// coincides with a fine node where its own basis function is 1, so the weight is
/// >= 1 and never zero, and each coarse value becomes a convex combination of the
/// fine values around it -- exact for a constant field, never amplifying, and
/// carrying the boundary data across.
static void RestrictSolution(const Operator &P, const Vector &fine,
                             Vector &coarse, MPI_Comm comm, int level)
{
   Vector ones(P.Height()), weights(P.Width());
   ones.UseDevice(true);
   weights.UseDevice(true);
   coarse.UseDevice(true);
   ones = 1.0;
   P.MultTranspose(ones, weights);
   P.MultTranspose(fine, coarse);
   coarse /= weights;

   if (!getenv("NEML2_MG_DEBUG")) { return; }

   // Device-aware throughout, and read back with explicit host reads: an earlier
   // version of this probe handed host scratch to a device operator and reported
   // uninitialized values alongside plausible ones.
   weights.HostRead();
   coarse.HostRead();
   real_t loc[4] = {fine.Normlinf(), coarse.Normlinf(), weights.Max(),
                    -weights.Min()};
   real_t glb[4];
   MPI_Allreduce(loc, glb, 4, MPITypeMap<real_t>::mpi_type, MPI_MAX, comm);

   int rank;
   MPI_Comm_rank(comm, &rank);
   if (rank != 0) { return; }
   std::cout << "    [mg] restrict L" << (level + 1) << " -> L" << level
             << ": |u_fine|inf=" << glb[0] << " |u_coarse|inf=" << glb[1]
             << " (x" << (glb[0] > 0 ? glb[1] / glb[0] : 0.0)
             << ")  weights in [" << -glb[3] << ", " << glb[2] << "]"
             << std::endl;
}

// I think we're going to have duplicate nonlinear forms on the fine level but maybe that's fine?
class NEML2Multigrid : public GeometricMultigrid
{
 private:
   std::shared_ptr<const ConstitutiveModel> constit;
   std::vector<MaterialStateManager *> level_states;
   real_t time;
   SolverConfig cfg;
   HypreBoomerAMG *amg = nullptr;
   PetscParMatrix *coarse_pmat = nullptr;   // GAMG coarse: assembled Mat (owned)
   PetscPreconditioner *coarse_pc = nullptr; // GAMG coarse PC (owned)
   std::vector<ParNonlinearForm> pnlfs;
   std::vector<Vector> coarser_solutions;
   const Vector &fine_solution;

 public:
   // Constructs a solid-mechanics multigrid for the ParFiniteElementSpaceHierarchy
   // and the array of essential boundaries. Each level evaluates the shared NEML2
   // model against its own persistent per-qp history store (level_states[level])
   // at the current load-step time.
   NEML2Multigrid(ParFiniteElementSpaceHierarchy &fespaces, Array<int> &ess_bdr,
                  std::shared_ptr<const ConstitutiveModel> constit_,
                  const std::vector<MaterialStateManager *> &level_states_,
                  real_t time_, const SolverConfig &cfg_,
                  const Vector &fine_solution_)
       : GeometricMultigrid(fespaces, ess_bdr), constit(constit_),
         level_states(level_states_), time(time_), cfg(cfg_),
         fine_solution(fine_solution_)
   {
      const auto num_levels = fespaces.GetNumLevels();
      const auto num_coarser_levels = num_levels - 1;

      // Build all the nonlinear forms first
      pnlfs.reserve(num_levels);
      for (int level = 0; level < fespaces.GetNumLevels(); ++level)
      {
         ConstructNonlinearForm(fespaces.GetFESpaceAtLevel(level), level);
      }
      MFEM_ASSERT(pnlfs.back().Height() == fine_solution.Size(),
                  "The size of the fine level nonlinear form should match the "
                  "size of our current solution");

      // Now we must construct the solution at the different levels
      coarser_solutions.resize(num_coarser_levels);
      if (num_coarser_levels)
      {
         auto create_coarser_solution = [this, &fespaces](
                                           const int level,
                                           const Vector &finer_solution)
         {
            auto &coarse_solution = coarser_solutions[level];
            coarse_solution.SetSize(pnlfs[level].Height());
            RestrictSolution(*fespaces.GetProlongationAtLevel(level),
                             finer_solution, coarse_solution,
                             fespaces.GetFESpaceAtLevel(level).GetComm(),
                             level);
         };
         create_coarser_solution(num_coarser_levels - 1, fine_solution);
         for (int level = num_coarser_levels - 2; level >= 0; --level)
         {
            create_coarser_solution(level, coarser_solutions[level + 1]);
         }
      }

      ConstructCoarseOperatorAndSolver(fespaces.GetFESpaceAtLevel(0));

      for (int level = 1; level < fespaces.GetNumLevels(); ++level)
      {
         ConstructOperatorAndSmoother(fespaces.GetFESpaceAtLevel(level), level);
      }

      // No longer need the gradient evaluation points
      coarser_solutions.clear();
   }

   ~NEML2Multigrid() override
   {
      delete amg;
      delete coarse_pc;   // must precede coarse_pmat (references it)
      delete coarse_pmat;
   }

 private:
   void ConstructNonlinearForm(ParFiniteElementSpace &fespace, int level)
   {
      auto &form = pnlfs.emplace_back(&fespace);
      if (level)
      {
         form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      else
      {
         form.SetAssemblyLevel(AssemblyLevel::FULL);
      }
      auto *const integ = new NEML2StressDivergenceIntegrator(constit, time);
      integ->SetState(level_states[level]);
      form.AddDomainIntegrator(integ);
      form.SetEssentialTrueDofs(*essentialTrueDofs[level]);
      form.Setup();
   }

   void ConstructCoarseOperatorAndSolver(ParFiniteElementSpace &coarse_fespace)
   {
      const auto *const coarse_pnlf = &pnlfs[0];
      const auto &coarse_solution = coarser_solutions.size() ? coarser_solutions[0]
                                                             : fine_solution;
      auto &coarse_operator = coarse_pnlf->GetGradient(coarse_solution);
      auto *const hypreCoarseMat = dynamic_cast<HypreParMatrix *>(&coarse_operator);
      MFEM_ASSERT(hypreCoarseMat,
                  "We should have created a parallel hypre csr matrix");

      if (cfg.coarse == "gamg")
      {
         // PETSc GAMG (smoothed aggregation) on the assembled coarse operator;
         // the fine level stays matrix-free. PC options come from the "coarse_"
         // prefix set in main (-coarse_pc_type gamg). One V-cycle as a fixed
         // (linear) coarse solve. The 6 rigid-body modes are attached as the
         // near-null-space so SA-AMG coarsens the elasticity operator well.
         coarse_pmat = new PetscParMatrix(hypreCoarseMat, Operator::PETSC_MATAIJ);
         if (cfg.near_null_space)
         {
            AttachRigidBodyNullSpace(coarse_fespace, *essentialTrueDofs[0],
                                     *coarse_pmat);
         }
         coarse_pc = new PetscPreconditioner(*coarse_pmat, "coarse_");
         AddLevel(hypreCoarseMat, coarse_pc, false, false);
         return;
      }

      MFEM_VERIFY(cfg.coarse == "boomeramg",
                  "coarse solver '" << cfg.coarse
                  << "' is not implemented (boomeramg | gamg | none)");

      amg = new HypreBoomerAMG(*hypreCoarseMat);
      amg->SetPrintLevel(-1);
      if (cfg.tuned_amg)
      {
         // Scalar BoomerAMG tuned for plasticity (ordering-agnostic). Elasticity
         // "systems" options would need Ordering::byVDIM; we run byNODES, so the
         // rigid-body near-null-space route is via GAMG instead (see benchmarks.md).
         HYPRE_Solver a = *amg;
         HYPRE_BoomerAMGSetStrongThreshold(a, 0.7);
         HYPRE_BoomerAMGSetCoarsenType(a, 8); // PMIS
         HYPRE_BoomerAMGSetInterpType(a, 6);  // ext+i
         HYPRE_BoomerAMGSetAggNumLevels(a, 4);
         HYPRE_BoomerAMGSetNumPaths(a, 2);
         HYPRE_BoomerAMGSetTruncFactor(a, 0.4);
      }

      if (cfg.coarse_mode == "vcycle")
      {
         // One AMG V-cycle as a fixed (linear) coarse solve, so a non-flexible
         // outer Krylov (CG/GMRES) stays valid. amg is owned by this object's
         // destructor, so it is not owned by the level.
         AddLevel(hypreCoarseMat, amg, false, false);
      }
      else // inner-cg: AMG-preconditioned CG (variable preconditioner)
      {
         CGSolver *pcg = new CGSolver(MPI_COMM_WORLD);
         pcg->SetPrintLevel(-1);
         pcg->SetMaxIter(10);
         pcg->SetRelTol(sqrt(1e-4));
         pcg->SetAbsTol(0.0);
         pcg->SetOperator(*hypreCoarseMat);
         pcg->SetPreconditioner(*amg);
         AddLevel(hypreCoarseMat, pcg, false, true);
      }
   }

   void ConstructOperatorAndSmoother(ParFiniteElementSpace &fespace, int level)
   {
      const auto *const pnlf = &pnlfs[level];
      const auto &level_soln = level == coarser_solutions.size() ? fine_solution
                                                                 : coarser_solutions[level];
      auto &opr = pnlf->GetGradient(level_soln);
      Vector diag(fespace.GetTrueVSize());
      opr.AssembleDiagonal(diag);

      // Matrix-free fine smoothers: Chebyshev (polynomial, degree smoother_order)
      // or a single damped point-Jacobi step. Both use only the diagonal +
      // mat-vecs, so the fine operator is never assembled.
      Solver *smoother;
      if (cfg.smoother == "jacobi")
      {
         smoother = new OperatorJacobiSmoother(diag, *essentialTrueDofs[level],
                                               2.0 / 3.0);
      }
      else
      {
         smoother = new OperatorChebyshevSmoother(opr, diag,
                                                  *essentialTrueDofs[level],
                                                  cfg.smoother_order,
                                                  fespace.GetParMesh()->GetComm());
      }

      AddLevel(&opr, smoother, false, true);
   }
};

// Single-level (no coarse grid) preconditioner: just the matrix-free fine
// smoother. Baseline for "what does the coarse grid buy" (combo #5, --coarse
// none). Owns its nonlinear form + smoother; the PETSc PCSHELL owns this object.
class SingleLevelSmoother : public Solver
{
 public:
   SingleLevelSmoother(ParFiniteElementSpace &fes, const Array<int> &ess_bdr,
                       std::shared_ptr<const ConstitutiveModel> constit,
                       MaterialStateManager *state, real_t time,
                       const SolverConfig &cfg, const Vector &soln)
       : Solver(fes.GetTrueVSize()), form(&fes)
   {
      fes.GetEssentialTrueDofs(const_cast<Array<int> &>(ess_bdr), ess_tdofs);
      form.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      auto *const integ = new NEML2StressDivergenceIntegrator(constit, time);
      integ->SetState(state);
      form.AddDomainIntegrator(integ);
      form.SetEssentialTrueDofs(ess_tdofs);
      form.Setup();
      Operator &grad = form.GetGradient(soln);
      Vector diag(fes.GetTrueVSize());
      grad.AssembleDiagonal(diag);
      if (cfg.smoother == "jacobi")
      {
         smoother.reset(new OperatorJacobiSmoother(diag, ess_tdofs, 2.0 / 3.0));
      }
      else
      {
         smoother.reset(new OperatorChebyshevSmoother(grad, diag, ess_tdofs,
                                                      cfg.smoother_order,
                                                      fes.GetParMesh()->GetComm()));
      }
   }

   void SetOperator(const Operator &) override {}
   void Mult(const Vector &x, Vector &y) const override { smoother->Mult(x, y); }

 private:
   ParNonlinearForm form;
   Array<int> ess_tdofs;
   std::unique_ptr<Solver> smoother;
};

class NEML2MultigridPreconditionerFactory : public PetscPreconditionerFactory
{
 public:
   NEML2MultigridPreconditionerFactory(ParFiniteElementSpaceHierarchy &fespaces_,
                                       Array<int> &ess_bdr_,
                                       std::shared_ptr<const ConstitutiveModel> constit_,
                                       const std::vector<MaterialStateManager *> &level_states_,
                                       const SolverConfig &cfg_,
                                       const Vector &fine_solution_)
       : PetscPreconditionerFactory(), fespaces(fespaces_), ess_bdr(ess_bdr_),
         constit(constit_), level_states(level_states_), cfg(cfg_),
         fine_solution(fine_solution_)
   {
   }

   /// Set the current load-step time, used by every subsequently rebuilt level.
   void SetTime(real_t t) { time = t; }

   // Since all the operator construction currently happens in the constructor,
   // let's just rebuild this every time. For a history-dependent model the
   // per-level tangents change from one Newton iterate to the next, so a rebuild
   // per NewPreconditioner is what keeps the multigrid consistent; each level
   // reuses its persistent history store (level_states) across rebuilds.
   mfem::Solver *NewPreconditioner(const mfem::OperatorHandle &) override
   {
      if (cfg.coarse == "none")
      {
         return new SingleLevelSmoother(fespaces.GetFinestFESpace(), ess_bdr,
                                        constit, level_states.back(), time, cfg,
                                        fine_solution);
      }
      return new NEML2Multigrid(fespaces, ess_bdr, constit, level_states, time,
                                cfg, fine_solution);
   }

   virtual ~NEML2MultigridPreconditionerFactory() = default;

 private:
   ParFiniteElementSpaceHierarchy &fespaces;
   Array<int> &ess_bdr;
   std::shared_ptr<const ConstitutiveModel> constit;
   std::vector<MaterialStateManager *> level_states;
   SolverConfig cfg;
   real_t time = 0.0;
   const Vector &fine_solution;
};

// Split a comma-separated option value, dropping empty entries.
static std::vector<std::string> SplitList(const std::string &s)
{
   std::vector<std::string> out;
   std::string item;
   std::istringstream stream(s);
   while (std::getline(stream, item, ','))
   {
      if (!item.empty()) { out.push_back(item); }
   }
   return out;
}

// Parse `--initial-conditions`: comma-separated `base=spec`, where spec is
// `identity` or a numeric constant applied to every component.
static std::map<std::string, InitialCondition>
ParseInitialConditions(const std::string &spec)
{
   std::map<std::string, InitialCondition> ics;
   for (const auto &entry : SplitList(spec))
   {
      const auto eq = entry.find('=');
      MFEM_VERIFY(eq != std::string::npos && eq > 0,
                  "Malformed initial condition '"
                     << entry << "'; expected <base>=<spec>");
      const std::string base = entry.substr(0, eq);
      const std::string value = entry.substr(eq + 1);
      ics[base] = (value == "identity") ? InitialCondition::Identity()
                                        : InitialCondition::Constant(
                                             std::stod(value));
   }
   return ics;
}

// Fill a per-quadrature-point orientation field for an ad hoc polycrystal.
//
// Grains come from a Voronoi tessellation of `num_grains` random seed points:
// every quadrature point takes the orientation of its nearest seed. That makes
// grain boundaries voxelized rather than conforming to element faces -- adequate
// to exercise the crystal plasticity coupling and grow a texture, but not a
// substitute for the conforming polycrystal mesher tracked in
// physics/cpfe/PLAN.md.
//
// Orientations are Modified Rodrigues Parameters (NEML2's `Rot`), drawn by
// sampling uniform unit quaternions (Shoemake) and mapping q -> MRP so the
// resulting texture is uniform over SO(3) rather than biased toward small
// rotations.
static void FillRandomGrainOrientations(const FiniteElementSpace &fes,
                                        int num_grains, int seed,
                                        ParameterFunction &orientation)
{
   Mesh &mesh = *fes.GetMesh();
   const int dim = mesh.Dimension();

   // Seed points in the mesh bounding box, plus one orientation per grain. Every
   // rank draws the same sequence from the same seed, so grains agree across
   // ranks and across multigrid levels without communication.
   std::mt19937 rng(seed);
   std::uniform_real_distribution<real_t> unit(0.0, 1.0);

   Vector lo, hi;
   mesh.GetBoundingBox(lo, hi);
   std::vector<Vector> seeds(num_grains, Vector(dim));
   for (auto &s : seeds)
   {
      for (int d = 0; d < dim; ++d) { s(d) = lo(d) + unit(rng) * (hi(d) - lo(d)); }
   }

   std::vector<std::array<real_t, 3>> mrp(num_grains);
   for (auto &r : mrp)
   {
      // Shoemake's uniform unit quaternion.
      const real_t u1 = unit(rng), u2 = unit(rng), u3 = unit(rng);
      const real_t s1 = std::sqrt(1 - u1), s2 = std::sqrt(u1);
      const real_t two_pi = 2 * M_PI;
      real_t q[4] = {s2 * std::cos(two_pi * u3), // w
                     s1 * std::sin(two_pi * u2), s1 * std::cos(two_pi * u2),
                     s2 * std::sin(two_pi * u3)};
      // MRP = q_vec / (1 + q_w); flip to the shadow parameters when q_w < 0 to
      // stay on the bounded branch (|MRP| <= 1).
      if (q[0] < 0)
      {
         for (auto &c : q) { c = -c; }
      }
      const real_t den = 1 + q[0];
      r = {q[1] / den, q[2] / den, q[3] / den};
   }

   const IntegrationRule &ir = NEML2IntRule(fes);
   const int nqp = ir.GetNPoints();
   real_t *data = orientation.HostWrite();
   Vector xq(dim);

   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      ElementTransformation &T = *mesh.GetElementTransformation(e);
      for (int p = 0; p < nqp; ++p)
      {
         T.SetIntPoint(&ir.IntPoint(p));
         T.Transform(ir.IntPoint(p), xq);
         int nearest = 0;
         real_t best = std::numeric_limits<real_t>::max();
         for (int g = 0; g < num_grains; ++g)
         {
            const real_t d2 = xq.DistanceSquaredTo(seeds[g]);
            if (d2 < best) { best = d2; nearest = g; }
         }
         real_t *dst = data + 3 * (e * nqp + p);
         for (int c = 0; c < 3; ++c) { dst[c] = mrp[nearest][c]; }
      }
   }
}

int main(int argc, char *argv[])
{
   // Initialize MPI and HYPRE
   Mpi::Init();
   Hypre::Init();
   int myid = Mpi::WorldRank();

   // NEML2 evaluates constitutive updates on small per-quadrature-point batches,
   // where torch's intra-op thread pool oversubscribes badly (it defaults to one
   // thread per core, and would also fight MPI ranks under mpirun). Parallelism
   // here comes from MPI domain decomposition, so pin torch to a single thread,
   // as the MOOSE-NEML2 coupling does. Must be set before any torch work.
   at::set_num_threads(1);
   at::set_num_interop_threads(1);

   // Parse command-line options
   const char *device_config = "cpu";
   constexpr auto petscrc_file = MFEM_SOURCE_DIR "/miniapps/neml2/"
                                                 "petscopts";
   int geometric_refinements = 0;
   int order_refinements = 1;
   int n = 5;
   std::string neml2_input = "elasticity_aoti/model";
   std::string neml2_model = "model";
   // Multi-step quasi-static loading + NEML2 variable-name configuration.
   int nt = 5;
   real_t dt = 0.01;
   real_t umax = 0.001;
   std::string kinematics = "small_strain";
   std::string kinematic_vars = "";
   std::string stress_var = "stress";
   std::string time_var = "t";
   std::string field_vars = "";
   std::string ic_spec = "";
   std::string orientation_var = "";
   int num_grains = 0;
   int grain_seed = 42;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&n, "-n", "--n",
                  "The number of elements in one dimension. The total number "
                  "will be a tensor product of this");
   args.AddOption(&geometric_refinements, "-gr", "--geometric-refinements",
                  "Number of geometric refinements done prior to order "
                  "refinements.");
   args.AddOption(&order_refinements, "-or", "--order-refinements",
                  "Number of order refinements. Finest level in the hierarchy "
                  "has order 2^{or}.");
   args.AddOption(&neml2_input, "-i", "--input",
                  "NEML2 cpp-aoti artifact to load, relative to "
                  "miniapps/neml2/: the folder produced by neml2-compile "
                  "(holding metadata.json + per-<device>/<dtype>/*.pt2), its "
                  "metadata.json, or the neml2-compile stub '.i'.");
   args.AddOption(&neml2_model, "-m", "--model",
                  "Name of the NEML2 model to use (only needed when -i is a "
                  "neml2-compile stub '.i').");
   args.AddOption(&nt, "-nt", "--num-time-steps",
                  "Number of quasi-static load steps.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time increment per load step (passed to the model as the "
                  "time variable when it declares one).");
   args.AddOption(&umax, "-umax", "--max-displacement",
                  "Total prescribed displacement at the final step; ramped "
                  "uniformly over the load steps.");
   args.AddOption(&kinematics, "-kin", "--kinematics",
                  "Kinematic measure the FE side feeds the model: small_strain "
                  "(sym(grad u), Mandel SR2; stress output is Cauchy) | "
                  "deformation_gradient (F = I + grad u, row-major R2; stress "
                  "output is PK2).");
   args.AddOption(&kinematic_vars, "-kv", "--kinematic-vars",
                  "Comma-separated NEML2 input names to drive with the "
                  "kinematic quantities, in order. Defaults to 'strain' for "
                  "small_strain and 'F' for deformation_gradient.");
   args.AddOption(&stress_var, "-yv", "--stress-var",
                  "Name of the stress output in the NEML2 model.");
   args.AddOption(&time_var, "-tv", "--time-var",
                  "Name of the time input in the NEML2 model (ignored if the "
                  "model declares no such input).");
   args.AddOption(&field_vars, "-fv", "--field-vars",
                  "Comma-separated NEML2 inputs supplied as per-quadrature-point "
                  "fields held fixed over the run (e.g. 'r' for crystal "
                  "orientation). Any declared input that is not kinematic, the "
                  "time, or ~k-lagged must be listed here.");
   args.AddOption(&ic_spec, "-ic", "--initial-conditions",
                  "Comma-separated <base>=<spec> initial conditions for state "
                  "variables, where <spec> is 'identity', or a numeric constant "
                  "applied to every component (e.g. 'Fp=identity,tauc=50'). "
                  "Unlisted state variables start at zero.");
   args.AddOption(&orientation_var, "-ov", "--orientation-var",
                  "Prescribed field (from --field-vars) to fill with a random "
                  "per-grain orientation. Requires --num-grains.");
   args.AddOption(&num_grains, "-ng", "--num-grains",
                  "Number of grains in the ad hoc polycrystal: element centers "
                  "are assigned to the nearest of this many random seed points "
                  "and each grain draws one random orientation. Grain "
                  "boundaries are voxelized, not conforming -- see "
                  "physics/cpfe/PLAN.md.");
   args.AddOption(&grain_seed, "-gs", "--grain-seed",
                  "RNG seed for grain seed points and orientations.");
   bool profile = false;
   args.AddOption(&profile, "-prof", "--profile", "-no-prof", "--no-profile",
                  "Report per-step constitutive vs linear-solve time and "
                  "iteration counts (adds device syncs around NEML2 calls).");
   // Solver/preconditioner selection (fine operator stays matrix-free). See
   // miniapps/neml2/benchmarks.md for the combinations under study.
   SolverConfig cfg;
   args.AddOption(&cfg.ksp, "-ksp", "--ksp",
                  "Outer Krylov method: cg | gmres | fgmres.");
   args.AddOption(&cfg.smoother, "-sm", "--smoother",
                  "Matrix-free fine smoother: chebyshev | jacobi.");
   args.AddOption(&cfg.smoother_order, "-smo", "--smoother-order",
                  "Chebyshev smoother degree.");
   args.AddOption(&cfg.coarse, "-co", "--coarse",
                  "Coarse solver: boomeramg | gamg | none.");
   args.AddOption(&cfg.coarse_mode, "-cm", "--coarse-mode",
                  "Coarse solve: inner-cg (AMG-preconditioned CG) | vcycle "
                  "(one AMG V-cycle, linear).");
   args.AddOption(&cfg.tuned_amg, "-tamg", "--tuned-amg", "-no-tamg",
                  "--no-tuned-amg",
                  "Apply plasticity-tuned scalar BoomerAMG parameters.");
   args.AddOption(&cfg.near_null_space, "-nns", "--near-null-space", "-no-nns",
                  "--no-near-null-space",
                  "Attach the 6 rigid-body modes as the GAMG coarse near-null-"
                  "space (helps elasticity; often neutral/worse for plasticity).");
   args.ParseCheck();

   // Enable hardware devices such as GPUs, and programming models such as CUDA
   Device device(device_config);
   if (Mpi::Root())
   {
      device.Print();
   }

   // Isolate NEML2 constitutive cost (return-map forward/jacobian) from linear
   // algebra so solver comparisons are not confounded by the plasticity cost.
   NEML2StressDivergenceIntegrator::SetProfiling(profile);

   MFEMInitializePetsc(nullptr, nullptr, petscrc_file, nullptr);

   // Outer Krylov selection (overrides the petscopts ksp_type). A large GMRES
   // restart favors robustness on the (possibly indefinite) plastic tangent.
   PetscCallAbort(PETSC_COMM_WORLD,
                  PetscOptionsSetValue(nullptr, "-ksp_type", cfg.ksp.c_str()));
   if (cfg.ksp == "gmres" || cfg.ksp == "fgmres")
   {
      PetscCallAbort(PETSC_COMM_WORLD,
                     PetscOptionsSetValue(nullptr, "-ksp_gmres_restart", "301"));
   }
   if (cfg.coarse == "gamg")
   {
      // GAMG on the assembled coarse operator (see NEML2Multigrid), "coarse_"
      // prefix. Smoothed aggregation is GAMG's default.
      PetscCallAbort(PETSC_COMM_WORLD,
                     PetscOptionsSetValue(nullptr, "-coarse_pc_type", "gamg"));
   }
   if (Mpi::Root() && cfg.coarse_mode == "inner-cg" && cfg.ksp != "fgmres")
   {
      std::cout << "Warning: --coarse-mode inner-cg is a variable preconditioner;"
                   " prefer --ksp fgmres or --coarse-mode vcycle for a consistent"
                   " outer Krylov.\n";
   }

   // Create a 3D mesh on the square domain [0,1]^3
   constexpr int dim = 3;
   Mesh mesh = Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON);

   // Define a parallel mesh
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   pmesh.SetCurvature(1);

   // Define a parallel finite element space on the parallel mesh
   auto *const fec = new H1_FECollection(1, /*dim=*/dim);
   auto *const coarse_fe_space = new ParFiniteElementSpace(&pmesh, fec,
                                                           /*vdim=*/dim,
                                                           Ordering::byNODES);
   Array<FiniteElementCollection *> collections;
   collections.Append(fec);
   ParFiniteElementSpaceHierarchy fespaces(&pmesh, coarse_fe_space, false,
                                           true);
   for (int level = 0; level < geometric_refinements; ++level)
   {
      fespaces.AddUniformlyRefinedLevel(dim, Ordering::byNODES);
   }
   for (int level = 0; level < order_refinements; ++level)
   {
      collections.Append(new H1_FECollection((int)std::pow(2, level + 1), dim));
      fespaces.AddOrderRefinedLevel(collections.Last(), dim, Ordering::byNODES);
   }
   auto &finest_fe_space = fespaces.GetFinestFESpace();

   HYPRE_BigInt size = finest_fe_space.GlobalTrueVSize();
   if (myid == 0)
   {
      std::cout << "Number of finite element unknowns: " << size << std::endl;
   }

   // The NEML2 constitutive model (v3 cpp-aoti route). The model is compiled
   // offline with `neml2-compile` into a self-describing artifact folder
   // (metadata.json + per-<device>/<dtype>/*.pt2). The `-i` option points at
   // that folder (or its metadata.json, or the neml2-compile stub `.i`). Device
   // and dtype are baked at compile time; here we only pick which compiled
   // <device> leaf to load and dispatch to.
   const std::string device_str = device.Allows(Backend::CUDA) ? "cuda" : "cpu";
   auto scheduler = std::make_shared<neml2::aoti::SimpleScheduler>(
      neml2::aoti::SimpleScheduler::Config{device_str, /*batch_size=*/0});

   const std::filesystem::path artifact_path =
      std::filesystem::path(MFEM_SOURCE_DIR) / "miniapps" / "neml2" / neml2_input;

   std::shared_ptr<neml2::aoti::DispatchedModel> cmodel;
   if (std::filesystem::is_directory(artifact_path))
   {
      cmodel = std::make_shared<neml2::aoti::DispatchedModel>(artifact_path,
                                                             scheduler);
   }
   else if (artifact_path.filename() == "metadata.json")
   {
      cmodel = std::make_shared<neml2::aoti::DispatchedModel>(
         artifact_path.parent_path(), scheduler);
   }
   else
   {
      cmodel = std::make_shared<neml2::aoti::DispatchedModel>(
         neml2::aoti::load_model(artifact_path, neml2_model, scheduler));
   }

   // Bind the driver's roles to this model's declared variable names. Everything
   // model-specific about the coupling lives in this struct, so one build drives
   // small-strain, finite-deformation and crystal plasticity models alike.
   VariableBinding binding;
   MFEM_VERIFY(kinematics == "small_strain" ||
                  kinematics == "deformation_gradient",
               "Unknown --kinematics '" << kinematics << "'");
   binding.mode = (kinematics == "deformation_gradient")
                     ? KinematicMode::DeformationGradient
                     : KinematicMode::SmallStrain;
   binding.kinematics = SplitList(kinematic_vars);
   if (binding.kinematics.empty())
   {
      binding.kinematics = {
         binding.mode == KinematicMode::DeformationGradient ? "F" : "strain"};
   }
   binding.stress = stress_var;
   binding.time = time_var;
   binding.fields = SplitList(field_vars);
   binding.initial_conditions = ParseInitialConditions(ic_spec);

   // Classifies the model's declared I/O into the FE-driven kinematic inputs,
   // the stress output, the time input, the prescribed per-point fields, and the
   // generic `~k`-lagged history variables. Shared (read-only) across the top
   // form and every multigrid level.
   auto constit = std::make_shared<const ConstitutiveModel>(cmodel, binding);
   if (myid == 0)
   {
      std::cout << "NEML2 model: kinematics(" << kinematics << ")=";
      for (const auto &v : constit->KinematicVars())
      {
         std::cout << '\'' << v.name << "'(vdim=" << v.vdim << ')';
      }
      std::cout << " stress='" << constit->StressName() << "' time='"
                << constit->TimeName() << "'";
      if (!constit->FieldVars().empty())
      {
         std::cout << " fields:";
         for (const auto &v : constit->FieldVars())
         {
            std::cout << ' ' << v.name << "(vdim=" << v.vdim << ')';
         }
      }
      std::cout << " history variables:";
      for (const auto &v : constit->StateVars())
      {
         std::cout << ' ' << v.var.base << "(vdim=" << v.var.vdim << ",~"
                   << v.depth << ')';
      }
      if (!constit->HasState())
      {
         std::cout << " (none, stateless)";
      }
      std::cout << std::endl;
   }

   // Persistent per-quadrature-point history store, one per mesh level. Owned
   // here so they survive the preconditioner factory's per-Newton rebuilds and
   // let coarse levels evolve their own history. The finest store is shared with
   // the top-level residual form.
   const int num_levels = fespaces.GetNumLevels();
   std::vector<std::unique_ptr<MaterialStateManager>> level_states;
   std::vector<MaterialStateManager *> level_state_ptrs;
   level_states.reserve(num_levels);
   for (int level = 0; level < num_levels; ++level)
   {
      const FiniteElementSpace &level_fes = fespaces.GetFESpaceAtLevel(level);
      level_states.push_back(std::make_unique<MaterialStateManager>(
         level_fes, constit->StateVars(), constit->FieldVars()));
      level_state_ptrs.push_back(level_states[level].get());

      // Seed the prescribed orientation field on this level's own quadrature
      // points. Each level resolves the same grain geometry at its own
      // resolution, so coarse levels see a consistent (if blockier) texture.
      if (!orientation_var.empty())
      {
         MFEM_VERIFY(num_grains > 0,
                     "--orientation-var requires --num-grains > 0");
         MFEM_VERIFY(level_states[level]->HasField(orientation_var),
                     "--orientation-var '"
                        << orientation_var
                        << "' is not among --field-vars, so the model does not "
                           "declare it as a prescribed input");
         FillRandomGrainOrientations(level_fes, num_grains, grain_seed,
                                     level_states[level]->Field(orientation_var));
      }
   }
   MaterialStateManager &fine_state = *level_states.back();

   // Essential boundary conditions (clamped BVP): x1 face fully fixed (u=0), x0
   // face prescribed (umax,0,0), ramped over the load steps. MFEM MakeCartesian3D
   // bdr attrs: x0=5, x1=3.
   Array<int> essential_bnd(pmesh.bdr_attributes.Max());
   Array<int> fixed_bnd(pmesh.bdr_attributes.Max());
   fixed_bnd = 0;
   fixed_bnd[2] = 1;
   essential_bnd = 0;
   essential_bnd[2] = 1;
   Vector uz(3);
   uz = 0.0;
   VectorConstantCoefficient zero_disp(uz);

   Array<int> displaced_bnd(pmesh.bdr_attributes.Max());
   displaced_bnd = 0;
   displaced_bnd[4] = 1;
   essential_bnd[4] = 1;
   Vector ug(3);
   ug = 0.0;

   // Initial condition
   ParGridFunction u(&finest_fe_space);
   u = 0;

   // Setup the parallel nonlinear form (top-level residual + matrix-free
   // Jacobian action). Shares the finest history store.
   ParNonlinearForm f(&finest_fe_space);
   const bool fully_assemble_jacobian = (geometric_refinements == 0) &&
                                        (order_refinements == 0);
   f.SetAssemblyLevel(fully_assemble_jacobian ? AssemblyLevel::FULL
                                              : AssemblyLevel::PARTIAL);
   auto *const top_integ = new NEML2StressDivergenceIntegrator(constit, 0.0);
   top_integ->SetState(&fine_state);
   f.AddDomainIntegrator(top_integ);
   Array<int> ess_tdof_list;
   finest_fe_space.GetEssentialTrueDofs(essential_bnd, ess_tdof_list);
   f.SetEssentialTrueDofs(ess_tdof_list);
   f.Setup();

   // Setup vector to solve for
   Vector &X = u.GetTrueVector();

   // Nonlinear solver
   auto * const newton = new PetscNonlinearSolver(MPI_COMM_WORLD, f);
   newton->SetAbsTol(1e-8);
   newton->SetRelTol(1e-6);
   newton->SetMaxIter(10);
   newton->SetPrintLevel(1);
   if (!fully_assemble_jacobian)
   {
      newton->SetJacobianType(Operator::PETSC_MATSHELL);
   }
   // Use the current state of u as the initial guess
   newton->iterative_mode = true;
   // Set multigrid preconditioner factory. MFEM PETSc wrapper code will try to
   // destroy this during the nonlinear solver destruction so allow them to do so
   auto *const pre_factory = new NEML2MultigridPreconditionerFactory(
      fespaces, essential_bnd, constit, level_state_ptrs, cfg, X);
   newton->SetPreconditionerFactory(pre_factory);

   // Prepare per-step ParaView output
   ParaViewDataCollection dc("neml2-output", finest_fe_space.GetParMesh());
   dc.RegisterField("disp", &u);

   // Quasi-static load stepping. Seed step 0 (all history zero).
   for (auto *state : level_state_ptrs)
   {
      state->Reset();
   }

   Vector R;                         // zero RHS for the nonlinear solve
   Vector res(X.Size());             // scratch residual for the commit

   // Solver introspection + running totals across the load steps.
   SNES snes = *newton; // mfem::petsc::SNES is ::SNES (same _p_SNES*)
   real_t tot_solve = 0.0, tot_res = 0.0, tot_tan = 0.0;
   long long tot_nl = 0, tot_lin = 0;

   for (int step = 1; step <= nt; ++step)
   {
      const real_t t_n = step * dt;

      // Propagate the step time to the top form and every multigrid level.
      top_integ->SetTime(t_n);
      pre_factory->SetTime(t_n);

      // Ramp the prescribed displacement, predicting the initial guess by scaling
      // the whole previous solution by the load ratio.
      //
      // The loading here is exactly proportional (one face fixed, the other
      // ramped), so this reproduces the new Dirichlet data exactly while keeping
      // the field smooth. Advancing only the boundary against a lagging interior
      // -- the obvious alternative -- opens a step-sized jump across the first
      // element at the loaded face, worth several times the physical strain: at
      // step 2 of `-nt 3 -umax 0.001` it puts max|F - I| at 4.1e-3 where the
      // converged step is 9.2e-4. That is a discretization artifact, but a local
      // return map under an n=8 power-law slip rule does not know that, and it
      // starts far enough out to exhaust its iteration budget.
      const real_t ux = umax * real_t(step) / real_t(nt);
      ug(0) = ux;
      VectorConstantCoefficient prescribed_disp(ug);
      u.SetFromTrueVector();
      if (step > 1) { u *= real_t(step) / real_t(step - 1); }
      u.ProjectBdrCoefficient(zero_disp, fixed_bnd);
      u.ProjectBdrCoefficient(prescribed_disp, displaced_bnd);
      u.SetTrueVector();

      if (myid == 0)
      {
         std::cout << "\n=== Load step " << step << "/" << nt << ", t = " << t_n
                   << ", u_x = " << ux << " ===" << std::endl;
      }

      NEML2StressDivergenceIntegrator::ResetConstitutiveTimers();
      StopWatch step_timer;
      step_timer.Start();
      newton->Mult(R, X);
      step_timer.Stop();
      MFEM_VERIFY(newton->GetConverged(),
                  "Newton failed to converge at load step " << step);

      // Per-step solver metrics (constitutive timers captured before the commit
      // eval so they reflect the solve only).
      const real_t solve_t = step_timer.RealTime();
      const real_t res_t =
         NEML2StressDivergenceIntegrator::ResidualConstitutiveTime();
      const real_t tan_t =
         NEML2StressDivergenceIntegrator::TangentConstitutiveTime();
      PetscInt nl_its = 0, lin_its = 0;
      PetscCallAbort(PETSC_COMM_WORLD, SNESGetIterationNumber(snes, &nl_its));
      PetscCallAbort(PETSC_COMM_WORLD, SNESGetLinearSolveIterations(snes, &lin_its));
      tot_solve += solve_t;
      tot_res += res_t;
      tot_tan += tan_t;
      tot_nl += static_cast<long long>(nl_its);
      tot_lin += static_cast<long long>(lin_its);
      if (myid == 0)
      {
         std::cout << "  Newton its=" << static_cast<long long>(nl_its)
                   << ", KSP its=" << static_cast<long long>(lin_its);
         if (profile)
         {
            std::cout << "  | solve=" << solve_t << "s, constitutive="
                      << (res_t + tan_t) << "s (residual=" << res_t
                      << ", tangent=" << tan_t << "), linear+other="
                      << (solve_t - res_t - tan_t) << "s";
         }
         std::cout << std::endl;
      }

      // Commit the converged step: refresh the finest history at X (stages the
      // converged strain + stress + internal state), set the step time on every
      // level, and advance current -> old everywhere.
      f.Mult(X, res);
      for (auto *state : level_state_ptrs)
      {
         if (constit->HasTime() && state->Has(constit->TimeName()))
         {
            state->Cur(constit->TimeName()) = t_n;
         }
         state->Advance();
      }

      // Report the committed history at every multigrid level (global L2 over
      // quadrature points) -- shows plastic accumulation / path dependence and
      // confirms that coarse levels evolve their own history, not just the fine.
      if (constit->HasState())
      {
         const MPI_Comm comm = finest_fe_space.GetComm();
         for (int lvl = 0; lvl < num_levels; ++lvl)
         {
            if (myid == 0)
            {
               std::cout << "  committed state L" << lvl << " (L2 over qp):";
            }
            for (const auto &v : constit->StateVars())
            {
               ParameterFunction &s = level_state_ptrs[lvl]->Old(v.var.base);
               const real_t nrm = std::sqrt(InnerProduct(comm, s, s));
               if (myid == 0) { std::cout << ' ' << v.var.base << '=' << nrm; }
            }
            if (myid == 0) { std::cout << std::endl; }
         }
      }

      // Save the solution in parallel using ParaView format
      u.SetFromTrueVector();
      dc.SetCycle(step);
      dc.SetTime(t_n);
      dc.Save();
   }

   if (myid == 0)
   {
      std::cout << "\n=== Totals over " << nt << " steps: Newton its=" << tot_nl
                << ", KSP its=" << tot_lin;
      if (profile)
      {
         std::cout << ", solve=" << tot_solve << "s, constitutive="
                   << (tot_res + tot_tan) << "s (residual=" << tot_res
                   << ", tangent=" << tot_tan << "), linear+other="
                   << (tot_solve - tot_res - tot_tan) << "s";
      }
      std::cout << " ===" << std::endl;
   }

   // The PETSc PCSHELL destroy hook frees only the generated preconditioner and
   // its context, not the factory itself, so delete the factory explicitly.
   delete newton;
   delete pre_factory;

   for (int level = 0; level < collections.Size(); ++level)
   {
      delete collections[level];
   }

   MFEMFinalizePetsc();

   return 0;
}
