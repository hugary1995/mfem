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

#include "operators.hpp"
#include <ATen/ATen.h>
#include <cmath>

namespace mfem
{

bool NEML2StressDivergenceIntegrator::s_profile = false;
real_t NEML2StressDivergenceIntegrator::s_residual_time = 0.0;
real_t NEML2StressDivergenceIntegrator::s_tangent_time = 0.0;

// Sync the device so host wall-clock timing captures async GPU work. No-op on CPU
// runs (and compiles to nothing when MFEM is built without a GPU backend).
static inline void DeviceSyncIfGPU()
{
#ifdef MFEM_DEVICE_SYNC
   if (Device::Allows(Backend::DEVICE_MASK)) { MFEM_DEVICE_SYNC; }
#endif
}

// NEML2's Mandel SR2 packing [xx, yy, zz, sqrt2 yz, sqrt2 xz, sqrt2 xy]: the flat
// index of component (i,j), and the sqrt(2) weight carried by each flat index.
static constexpr int kMandel[3][3] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};
static inline real_t MandelWeight(int a)
{
   return a < 3 ? real_t(1) : std::sqrt(real_t(2));
}

// Expand a batched Mandel SR2 (*B,6) into a full symmetric (*B,3,3).
static at::Tensor MandelVecToFull(const at::Tensor &v)
{
   using namespace at::indexing;
   std::vector<int64_t> shape(v.sizes().begin(), v.sizes().end() - 1);
   shape.insert(shape.end(), {3, 3});
   at::Tensor out = at::empty(shape, v.options());
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
         const int a = kMandel[i][j];
         out.index_put_({Ellipsis, i, j}, v.index({Ellipsis, a}) / MandelWeight(a));
      }
   return out;
}

// Expand the leading Mandel index of a batched dS/dF block (*B,6,3,3) into a
// symmetric (M,J) pair, giving A_MJkL = dS_MJ/dF_kL with shape (*B,3,3,3,3).
static at::Tensor MandelRowsToFull(const at::Tensor &b)
{
   using namespace at::indexing;
   std::vector<int64_t> shape(b.sizes().begin(), b.sizes().end() - 3);
   shape.insert(shape.end(), {3, 3, 3, 3});
   at::Tensor out = at::empty(shape, b.options());
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
         const int a = kMandel[i][j];
         out.index_put_({Ellipsis, i, j, Slice(), Slice()},
                        b.index({Ellipsis, a, Slice(), Slice()}) /
                           MandelWeight(a));
      }
   return out;
}

// Assemble the total-Lagrangian tangent D_iJkL = dP_iJ/dF_kL from the model's
// material block, splitting into the two standard contributions:
//
//   geometric: delta_ik S_JL          (stress carried along by the rotation and
//                                      stretch of the reference gradient)
//   material:  F_iM dS_MJ/dF_kL       (the constitutive response proper)
//
// The result is non-symmetric in general, which is why the assembled kernels
// contract it in an explicit index order rather than exploiting major symmetry.
static at::Tensor DeformationGradientTangent(const at::Tensor &dSdF,
                                             const at::Tensor &S,
                                             const at::Tensor &F)
{
   MFEM_VERIFY(dSdF.dim() == 4,
               "Expected a batched dS/dF block of shape (nqp,6,3,3), got dim "
                  << dSdF.dim());
   const at::Tensor A = MandelRowsToFull(dSdF);            // (nqp,3,3,3,3)
   const at::Tensor S_full = MandelVecToFull(S);           // (nqp,3,3)
   const at::Tensor eye = at::eye(3, F.options());
   const at::Tensor geometric = at::einsum("ik,bjl->bijkl", {eye, S_full});
   const at::Tensor material = at::einsum("bim,bmjkl->bijkl", {F, A});
   return (geometric + material).contiguous();
}

NEML2StressDivergenceIntegrator::NEML2StressDivergenceIntegrator(
   std::shared_ptr<const ConstitutiveModel> constit, real_t time,
   const IntegrationRule *ir)
    : StressDivergenceIntegrator<NonlinearFormIntegrator>(ir), _t(time),
      _constit_op(std::move(constit))
{
   _mode = _constit_op->Mode();
   const auto &kin = _constit_op->KinematicVars();
   if (_mode == KinematicMode::SmallStrain)
   {
      MFEM_VERIFY(kin.size() == 1 && kin[0].vdim == 6,
                  "Small-strain kinematics bind exactly one SR2 input; the model "
                  "declares "
                     << kin.size() << " kinematic input(s)");
   }
   else
   {
      MFEM_VERIFY(kin.size() == 1 && kin[0].vdim == 9,
                  "Deformation-gradient kinematics bind exactly one R2 input; the "
                  "model declares "
                     << kin.size() << " kinematic input(s)");
   }
}

void NEML2StressDivergenceIntegrator::AssemblePA(const FiniteElementSpace &fe_space)
{
   if (fespace)
   {
      MFEM_ASSERT(fespace == &fe_space,
                  "We're assembling with a different finite element space?");
      // We're already partially assembled
      return;
   }

   StressDivergenceIntegrator<NonlinearFormIntegrator>::AssemblePA(fe_space);
   if (this->vdim < 2)
   {
      mfem_error("NEML2StressDivergenceIntegrator is only meant to be used in "
                 "multiple dimensions");
   }

   _ordering = fe_space.GetOrdering();
   auto *const mesh = fe_space.GetMesh();

   _q_space_symr2 = std::make_unique<UniformParameterSpace>(*mesh,
                                                            *this->IntRule, 6);
   _q_space_r2 = std::make_unique<UniformParameterSpace>(*mesh, *this->IntRule, 9);

   _stress = std::make_unique<ParameterFunction>(*_q_space_symr2);
   _pk = std::make_unique<ParameterFunction>(*_q_space_r2);
   _gradu = std::make_unique<ParameterFunction>(*_q_space_r2);
   _stress->UseDevice(true);
   _pk->UseDevice(true);
   _gradu->UseDevice(true);

   // One storage slot per kinematic input the model declares, sized by its base
   // shape, for both the residual point and the linearization point.
   for (const auto &kv : _constit_op->KinematicVars())
   {
      _kin_spaces.push_back(
         std::make_unique<UniformParameterSpace>(*mesh, *this->IntRule, kv.vdim));
      auto &space = *_kin_spaces.back();
      _kin.push_back(std::make_unique<ParameterFunction>(space));
      _kin_lin.push_back(std::make_unique<ParameterFunction>(space));
      _kin.back()->UseDevice(true);
      _kin_lin.back()->UseDevice(true);
      _kin_ptrs.push_back(_kin.back().get());
      _kin_lin_ptrs.push_back(_kin_lin.back().get());
   }
}

template <int vdim>
void NEML2StressDivergenceIntegrator::ComputeStrainImpl(const Vector &x,
                                                        ParameterFunction &strain)
                                                                                const
{
   using future::tensor;

   constexpr int d = vdim;

   // Assuming all elements are the same
   const QuadratureInterpolator *E_To_Q_Map = this->fespace->GetQuadratureInterpolator(
                                                                                   *this->IntRule);
   E_To_Q_Map->SetOutputLayout(_ordering == Ordering::byNODES ? QVectorLayout::byNODES
                                                              : QVectorLayout::byVDIM);
   // interpolate physical derivatives to quadrature points.
   E_To_Q_Map->PhysDerivatives(x, *this->q_vec);

   const int numPoints = this->IntRule->GetNPoints();
   const int numEls = this->fespace->GetNE();
   const auto Q = Reshape(this->q_vec->Read(), numPoints, d, d, numEls);
   // device strain
   auto dStrain = Reshape(strain.Write(), 6, numPoints, numEls);
   mfem::forall_2D(numEls, numPoints, 1,
                   [=] MFEM_HOST_DEVICE(int e)
                   {
                      MFEM_FOREACH_THREAD(p, x, numPoints)
                      {
                         tensor<real_t, d, d> dudx;
                         // load grad(x) into dudx
                         for (int j = 0; j < d; j++)
                         {
                            for (int i = 0; i < d; i++)
                            {
                               dudx(i, j) = Q(p, i, j, e);
                            }
                         }
                         const auto epsilon = real_t(0.5) *
                                              (dudx + transpose(dudx));
                         // NEML2 uses Mandel notation for symmetric 2nd order tensors.
                         constexpr real_t sqrt2 = 1.4142135623730951_r;
                         dStrain(0, p, e) = epsilon(0, 0);
                         dStrain(1, p, e) = epsilon(1, 1);
                         dStrain(5, p, e) = sqrt2 * epsilon(0, 1);
                         if (d == 2) // NEML2 always expects 3D
                         {
                            dStrain(2, p, e) = 0;
                            dStrain(3, p, e) = 0;
                            dStrain(4, p, e) = 0;
                         }
                         else
                         {
                            dStrain(2, p, e) = epsilon(2, 2);
                            dStrain(3, p, e) = sqrt2 * epsilon(1, 2);
                            dStrain(4, p, e) = sqrt2 * epsilon(0, 2);
                         }
                      }
                   });
}

template <int vdim>
void NEML2StressDivergenceIntegrator::ComputeGradUImpl(const Vector &x,
                                                       bool add_identity,
                                                       ParameterFunction &gradu)
                                                                                const
{
   constexpr int d = vdim;

   // Assuming all elements are the same
   const QuadratureInterpolator *E_To_Q_Map = this->fespace->GetQuadratureInterpolator(
                                                                                   *this->IntRule);
   E_To_Q_Map->SetOutputLayout(_ordering == Ordering::byNODES ? QVectorLayout::byNODES
                                                              : QVectorLayout::byVDIM);
   // The mesh is never moved, so "physical" derivatives are derivatives with
   // respect to the reference configuration -- exactly what a total-Lagrangian
   // deformation gradient needs.
   E_To_Q_Map->PhysDerivatives(x, *this->q_vec);

   const int numPoints = this->IntRule->GetNPoints();
   const int numEls = this->fespace->GetNE();
   const auto Q = Reshape(this->q_vec->Read(), numPoints, d, d, numEls);
   auto G = Reshape(gradu.Write(), 9, numPoints, numEls);
   mfem::forall_2D(numEls, numPoints, 1,
                   [=] MFEM_HOST_DEVICE(int e)
                   {
                      MFEM_FOREACH_THREAD(p, x, numPoints)
                      {
                         // Row-major 3x3, zero-padded out of plane in 2D so the
                         // (always 3D) NEML2 model sees plane strain.
                         for (int k = 0; k < 9; ++k) { G(k, p, e) = 0; }
                         for (int i = 0; i < d; ++i)
                         {
                            for (int j = 0; j < d; ++j)
                            {
                               G(3 * i + j, p, e) = Q(p, i, j, e);
                            }
                         }
                         if (add_identity)
                         {
                            G(0, p, e) += 1;
                            G(4, p, e) += 1;
                            G(8, p, e) += 1;
                         }
                      }
                   });
}

void NEML2StressDivergenceIntegrator::ComputeGradU(const Vector &X,
                                                   bool add_identity,
                                                   ParameterFunction &gradu) const
{
   if (this->vdim == 2)
   {
      this->ComputeGradUImpl<2>(X, add_identity, gradu);
   }
   else if (this->vdim == 3)
   {
      this->ComputeGradUImpl<3>(X, add_identity, gradu);
   }
}

void NEML2StressDivergenceIntegrator::ComputeKinematics(
   const Vector &X, std::vector<std::unique_ptr<ParameterFunction>> &kin) const
{
   if (_mode == KinematicMode::SmallStrain)
   {
      if (this->vdim == 2) { this->ComputeStrainImpl<2>(X, *kin[0]); }
      else if (this->vdim == 3) { this->ComputeStrainImpl<3>(X, *kin[0]); }
      return;
   }
   this->ComputeGradU(X, /*add_identity=*/true, *kin[0]);
}

void NEML2StressDivergenceIntegrator::ComputePK(
   const std::vector<std::unique_ptr<ParameterFunction>> &kin) const
{
   const int npts = _stress->Size() / 6;
   const auto S = Reshape(_stress->Read(), 6, npts);
   auto T = Reshape(_pk->Write(), 9, npts);
   const bool finite_deformation = (_mode == KinematicMode::DeformationGradient);
   // A deformation gradient is present only in the finite-deformation branch;
   // alias the stress otherwise so the (unused) pointer is still valid.
   const auto F = Reshape(
      finite_deformation ? kin[0]->Read() : _stress->Read(), 9, npts);

   mfem::forall(npts,
                [=] MFEM_HOST_DEVICE(int p)
                {
                   constexpr real_t sqrt2 = 1.4142135623730951_r;
                   // Mandel SR2 -> row-major 3x3.
                   real_t Sf[9];
                   Sf[0] = S(0, p);
                   Sf[4] = S(1, p);
                   Sf[8] = S(2, p);
                   Sf[5] = Sf[7] = S(3, p) / sqrt2;
                   Sf[2] = Sf[6] = S(4, p) / sqrt2;
                   Sf[1] = Sf[3] = S(5, p) / sqrt2;

                   if (!finite_deformation)
                   {
                      // Small strain: the model's Cauchy stress is already the
                      // stress conjugate to the reference gradient.
                      for (int k = 0; k < 9; ++k) { T(k, p) = Sf[k]; }
                      return;
                   }
                   // P = F S
                   for (int i = 0; i < 3; ++i)
                   {
                      for (int j = 0; j < 3; ++j)
                      {
                         real_t sum = 0;
                         for (int m = 0; m < 3; ++m)
                         {
                            sum += F(3 * i + m, p) * Sf[3 * m + j];
                         }
                         T(3 * i + j, p) = sum;
                      }
                   }
                });
}

template <int vdim>
void NEML2StressDivergenceIntegrator::ComputeDivergenceImpl(
   const ParameterFunction &pk, Vector &R) const
{
   using future::det;
   using future::inv;
   using future::make_tensor;
   using future::tensor;

   constexpr int d = vdim;

   const int numPoints = this->IntRule->GetNPoints();
   const int numEls = this->fespace->GetNE();
   auto Q = Reshape(this->q_vec->Write(), numPoints, d, d, numEls);
   const auto dPK = Reshape(pk.Read(), 9, numPoints, numEls);
   const auto J = Reshape(this->geom->J.Read(), numPoints, d, d, numEls);

   const real_t *ipWeights = this->IntRule->GetWeights().Read();
   mfem::forall_2D(numEls, numPoints, 1,
                   [=] MFEM_HOST_DEVICE(int e)
                   {
                      MFEM_FOREACH_THREAD(p, x, numPoints)
                      {
                         // clang-format off
                           const auto invJ = inv(make_tensor<d, d>([&](int i, int j)
                                                                  { return J(p, i, j, e); }));
                         // clang-format on
                         tensor<real_t, d, d> stress_tensor;
                         for (int i = 0; i < d; ++i)
                         {
                            for (int j = 0; j < d; ++j)
                            {
                               stress_tensor(i, j) = dPK(3 * i + j, p, e);
                            }
                         }
                         const auto JxW = ipWeights[p] / det(invJ);
                         // Fold the inverse Jacobian into the stress so the
                         // reduction below can contract against reference-frame
                         // shape function gradients directly.
                         const auto sigma_ref_weighted = stress_tensor *
                                                         transpose(invJ) * JxW;
                         for (int m = 0; m < d; ++m)
                         {
                            for (int q = 0; q < d; ++q)
                            {
                               Q(p, m, q, e) = sigma_ref_weighted(q, m);
                            }
                         }
                      }
                   });

   // Reduce quadrature function to an E-Vector
   const auto QRead = Reshape(this->q_vec->Read(), numPoints, d, d, numEls);
   const auto G = Reshape(this->maps->G.Read(), numPoints, d, this->ndofs);
   const auto nDofs = this->ndofs;
   auto rDev = Reshape(R.ReadWrite(), this->ndofs, d, numEls);
   mfem::forall_2D(numEls, d, nDofs,
                   [=] MFEM_HOST_DEVICE(int e)
                   {
                      MFEM_FOREACH_THREAD(i, y, nDofs)
                      {
                         MFEM_FOREACH_THREAD(q, x, d)
                         {
                            real_t sum = 0.;
                            for (int m = 0; m < d; m++)
                            {
                               for (int p = 0; p < numPoints; p++)
                               {
                                  sum += QRead(p, m, q, e) * G(p, m, i);
                               }
                            }
                            rDev(i, q, e) += sum;
                         }
                      }
                   });
}

void NEML2StressDivergenceIntegrator::ComputeDivergence(const ParameterFunction &pk,
                                                        Vector &R) const
{
   if (this->vdim == 2)
   {
      this->ComputeDivergenceImpl<2>(pk, R);
   }
   else if (this->vdim == 3)
   {
      this->ComputeDivergenceImpl<3>(pk, R);
   }
}

// Diagnostic (NEML2_MG_DEBUG=1): the kinematic range the model is about to be
// evaluated at. `nqp` identifies the multigrid level, since each level carries
// its own quadrature count. Worth having because a NEML2 return-map failure
// aborts the process without saying what it was handed, and the interesting
// question is almost always whether that input was physical.
static void ReportKinematics(const char *what, real_t t,
                             const std::vector<ParameterFunction *> &kin,
                             const ConstitutiveModel &constit)
{
   if (!getenv("NEML2_MG_DEBUG") || Mpi::WorldRank() != 0) { return; }
   const auto opts = constit.Options();
   for (size_t i = 0; i < kin.size(); ++i)
   {
      const at::Tensor v = ConstitutiveModel::Wrap(opts, *kin[i]);
      const at::Tensor dev =
         constit.Mode() == KinematicMode::DeformationGradient
            ? v.reshape({-1, 3, 3}) - at::eye(3, opts)
            : v;
      std::cout << "    [kin] " << what << " t=" << t << " nqp=" << v.size(0)
                << " max|" << constit.KinematicVars()[i].name
                << " dev|=" << dev.abs().max().item<double>() << std::endl;
   }
}

void NEML2StressDivergenceIntegrator::AddMultPA(const Vector &X,
                                                Vector &R) const
{
   MFEM_VERIFY(_state, "NEML2StressDivergenceIntegrator: SetState() not called");

   // displacement -> kinematics
   this->ComputeKinematics(X, _kin);
   ReportKinematics("residual", _t, _kin_ptrs, *_constit_op);

   // kinematics -> stress via NEML2 (reads old history, stages new state)
   if (s_profile)
   {
      StopWatch sw;
      DeviceSyncIfGPU();
      sw.Start();
      _constit_op->Mult(_kin_ptrs, *_stress, _t, *_state);
      DeviceSyncIfGPU();
      sw.Stop();
      s_residual_time += sw.RealTime();
   }
   else
   {
      _constit_op->Mult(_kin_ptrs, *_stress, _t, *_state);
    }

   // stress -> residuals
   this->ComputePK(_kin);
   this->ComputeDivergence(*_pk, R);
}

void NEML2StressDivergenceIntegrator::AssembleGradPA(const Vector &X,
                                                     const FiniteElementSpace &fes)
{
   // Make sure our basis functions, geometric factors, and other data is already initialized
   this->AssemblePA(fes);

   MFEM_VERIFY(_state, "NEML2StressDivergenceIntegrator: SetState() not called");

   // Evaluate the consistent tangent once at this linearization point and cache
   // it in two forms sharing the single NEML2 solve: the full D_iJkL (used by the
   // assembled paths: PA diagonal, element assembly, coarse matrix) and its
   // (9,9) flattening (used by the matrix-free gradient action).
   this->ComputeKinematics(X, _kin_lin);
   ReportKinematics("tangent ", _t, _kin_lin_ptrs, *_constit_op);
   std::vector<at::Tensor> blocks;
   if (s_profile)
   {
      StopWatch sw;
      DeviceSyncIfGPU();
      sw.Start();
      _constit_op->Tangent(_kin_lin_ptrs, blocks, *_stress, _t, *_state);
      DeviceSyncIfGPU();
      sw.Stop();
      s_tangent_time += sw.RealTime();
   }
   else
   {
      _constit_op->Tangent(_kin_lin_ptrs, blocks, *_stress, _t, *_state);
   }

   if (_mode == KinematicMode::SmallStrain)
   {
      // d(sigma)/d(grad u) is the small-strain stiffness itself: the symmetrizing
      // half-factors sum away against C's minor symmetry.
      _tangent = ConstitutiveModel::MandelToFullStiffness(blocks[0]);
   }
   else
   {
      const auto opts = _constit_op->Options();
      const at::Tensor S = ConstitutiveModel::Wrap(opts, *_stress);
      const at::Tensor F = ConstitutiveModel::Wrap(opts, *_kin_lin[0])
                              .reshape({-1, 3, 3});
      _tangent = DeformationGradientTangent(blocks[0], S, F);
   }

   const at::Tensor &D = _tangent.value();
   _flat_tangent = (D.dim() > 4) ? D.reshape({D.size(0), 9, 9})
                                 : D.reshape({9, 9});
}

template <int vdim>
void NEML2StressDivergenceIntegrator::AssembleGradDiagonalPAImpl(Vector &diag) const
{
   using future::det;
   using future::inv;
   using future::make_tensor;
   using future::tensor;

   // Assuming all elements are the same
   static constexpr int d = vdim;
   const auto numPoints = this->IntRule->GetNPoints();
   const auto numEls = this->fespace->GetNE();
   const auto nDofs = this->ndofs;
   const auto J = Reshape(this->geom->J.Read(), numPoints, d, d, numEls);
   const auto G = Reshape(this->maps->G.Read(), numPoints, d, nDofs);
   auto diagDev = Reshape(diag.Write(), nDofs, d, numEls);
   const real_t *ipWeights = this->IntRule->GetWeights().Read();
   // The tangent D_iJkL; unbatched (dim()==4) when the derivative is
   // batch-independent (e.g. linear elasticity).
   const at::Tensor &full_tangent = _tangent.value();
   const bool constant_tangent = (full_tangent.dim() == 4);
   const int tangent_qp_size = constant_tangent ? 1 : numPoints;
   const int tangent_elem_size = constant_tangent ? 1 : numEls;
   // MFEM's Reshape indexes leftmost-fastest while the torch tensor is row-major,
   // so the four tensor slots appear reversed: Craw(L,k,J,i) is D_iJkL.
   const auto Craw = Reshape(full_tangent.data_ptr<real_t>(), 3, 3, 3, 3,
                             tangent_qp_size, tangent_elem_size);

   // clang-format off
   mfem::forall_2D(numEls, nDofs, d,
                   [=] MFEM_HOST_DEVICE(int e)
                   {
                      const int tangent_e_index = constant_tangent ? 0 : e;
                      MFEM_FOREACH_THREAD(ic, y, d)
                      {
                        MFEM_FOREACH_THREAD(IScalar, x, nDofs)
                        {
                           real_t sum = 0;
                           for (int p = 0; p < numPoints; ++p)
                           {
                               const int tangent_p_index = constant_tangent ? 0
                                                                            : p;
                               const auto invJ = inv(make_tensor<d, d>([&](int i,int j){
                                 return J(p,i,j,e);}));

                               const real_t w = ipWeights[p] / det(invJ);

                               // Compute shape function gradients
                               real_t dphiI[d] = {0};
                               for (int alpha = 0; alpha < d;
                                    ++alpha) // reference coord
                               {
                                  const auto gI = G(p, alpha, IScalar);
                                  for (int m = 0; m < d; ++m) // physical coord
                                  {
                                     const auto jac_map = invJ(alpha, m);
                                     dphiI[m] += gI * jac_map;
                                  }
                               }

                               // K_(I,ic)(I,ic) = dphiI_J D_(ic)J(ic)L dphiI_L.
                               // The diagonal is the I==J, ic==jc case of the
                               // element matrix below.
                               real_t val = 0.;
                               for (int Jd = 0; Jd < d; ++Jd)
                               {
                                  for (int L = 0; L < d; ++L)
                                  {
                                     val += dphiI[Jd] *
                                            Craw(L, ic, Jd, ic,
                                                 tangent_p_index,
                                                 tangent_e_index) *
                                            dphiI[L];
                                  }
                               }
                               sum += w * val;
                            }
                            diagDev(IScalar, ic, e) = sum;
                         }
                        }
                      });
   // clang-format on
}

void NEML2StressDivergenceIntegrator::AssembleGradDiagonalPA(Vector &diag) const
{
   if (this->vdim == 2)
   {
      this->AssembleGradDiagonalPAImpl<2>(diag);
   }
   else
   {
      this->AssembleGradDiagonalPAImpl<3>(diag);
   }
}

template <int vdim>
void NEML2StressDivergenceIntegrator::AssembleGradEAImpl(Vector &emat)
{
   using future::det;
   using future::inv;
   using future::make_tensor;
   using future::tensor;

   // Assuming all elements are the same
   static constexpr int d = vdim;
   const auto numPoints = this->IntRule->GetNPoints();
   const auto numEls = this->fespace->GetNE();
   const auto nDofs = this->ndofs;
   const auto vDofs = d * nDofs;
   const auto J = Reshape(this->geom->J.Read(), numPoints, d, d, numEls);
   const auto G = Reshape(this->maps->G.Read(), numPoints, d, nDofs);
   auto ematDev = Reshape(emat.Write(), vDofs, vDofs, numEls);
   const real_t *ipWeights = this->IntRule->GetWeights().Read();
   // The tangent D_iJkL; unbatched (dim()==4) when the derivative is
   // batch-independent (e.g. linear elasticity).
   const at::Tensor &full_tangent = _tangent.value();
   const bool constant_tangent = (full_tangent.dim() == 4);
   const int tangent_qp_size = constant_tangent ? 1 : numPoints;
   const int tangent_elem_size = constant_tangent ? 1 : numEls;
   // MFEM's Reshape indexes leftmost-fastest while the torch tensor is row-major,
   // so the four tensor slots appear reversed: Craw(L,k,J,i) is D_iJkL.
   const auto Craw = Reshape(full_tangent.data_ptr<real_t>(), 3, 3, 3, 3,
                             tangent_qp_size, tangent_elem_size);

   // clang-format off
   mfem::forall_2D(numEls, vDofs, vDofs,
                   [=] MFEM_HOST_DEVICE(int e)
                   {
                      const int tangent_e_index = constant_tangent ? 0 : e;
                      MFEM_FOREACH_THREAD(JVec, y, vDofs)
                      {
                         MFEM_FOREACH_THREAD(IVec, x, vDofs)
                         {
                            // We can't just pick an ordering because upstream code always varies by
                            // node index most quickly, then by vdim, and then by ne for our element
                            // assembly data
                            const int ic = IVec / nDofs;
                            const int IScalar = IVec % nDofs;
                            const int jc = JVec / nDofs;
                            const int JScalar = JVec % nDofs;

                            real_t sum = 0;

                            for (int p = 0; p < numPoints; p++)
                            {
                               const int tangent_p_index = constant_tangent ? 0
                                                                            : p;
                               const auto invJ = inv(make_tensor<d, d>([&](int i,int j){
                                 return J(p,i,j,e);}));

                               const real_t w = ipWeights[p] / det(invJ);

                               // Compute shape function gradients
                               real_t dphiI[d] = {0}, dphiJ[d] = {0};
                               for (int alpha = 0; alpha < d;
                                    ++alpha) // reference coord
                               {
                                  const auto gI = G(p, alpha, IScalar);
                                  const auto gJ = G(p, alpha, JScalar);
                                  for (int m = 0; m < d; ++m) // physical coord
                                  {
                                     const auto jac_map = invJ(alpha, m);
                                     dphiI[m] += gI * jac_map;
                                     dphiJ[m] += gJ * jac_map;
                                  }
                               }

                               // K_(I,ic)(J,jc) = dphiI_Jd D_(ic)(Jd)(jc)L dphiJ_L.
                               // Contracted in this explicit order because the
                               // finite-deformation tangent has neither minor nor
                               // major symmetry to fall back on.
                               real_t val = 0.;
                               for (int Jd = 0; Jd < d; ++Jd)
                               {
                                  for (int L = 0; L < d; ++L)
                                  {
                                     val += dphiI[Jd] *
                                            Craw(L, jc, Jd, ic,
                                                 tangent_p_index,
                                                 tangent_e_index) *
                                            dphiJ[L];
                                  }
                               }
                               sum += w * val;
                            }
                            ematDev(IVec, JVec, e) = sum;
                         }
                      }
                   });
   // clang-format on
}

void NEML2StressDivergenceIntegrator::AssembleGradEA(const Vector &X,
                                                     const FiniteElementSpace &fes,
                                                     Vector &emat)
{
   this->AssembleGradPA(X, fes);
   if (this->vdim == 2)
   {
      this->AssembleGradEAImpl<2>(emat);
   }
   else if (this->vdim == 3)
   {
      this->AssembleGradEAImpl<3>(emat);
   }
}

void NEML2StressDivergenceIntegrator::AddMultGradPA(const Vector &dX,
                                                    Vector &dR) const
{
   MFEM_VERIFY(_flat_tangent.has_value(),
               "AssembleGradPA must run before AddMultGradPA");

   // grad(du) at the quadrature points (no identity: this is an increment)
   this->ComputeGradU(dX, /*add_identity=*/false, *_gradu);

   // dT = D : grad(du), the cached consistent-tangent action as a per-point 9x9
   // matvec -- no constitutive re-evaluation.
   _constit_op->ApplyStoredTangent(_flat_tangent.value(), *_gradu, *_pk);

   // dR = div(dT)
   this->ComputeDivergence(*_pk, dR);
}

template void
NEML2StressDivergenceIntegrator::ComputeStrainImpl<2>(const Vector &x,
                                                      ParameterFunction &strain)
                                                                                const;
template void
NEML2StressDivergenceIntegrator::ComputeStrainImpl<3>(const Vector &x,
                                                      ParameterFunction &strain)
                                                                                const;
template void
NEML2StressDivergenceIntegrator::ComputeGradUImpl<2>(const Vector &x,
                                                     bool add_identity,
                                                     ParameterFunction &gradu) const;
template void
NEML2StressDivergenceIntegrator::ComputeGradUImpl<3>(const Vector &x,
                                                     bool add_identity,
                                                     ParameterFunction &gradu) const;
template void
NEML2StressDivergenceIntegrator::ComputeDivergenceImpl<2>(const ParameterFunction &pk,
                                                          Vector &R) const;
template void
NEML2StressDivergenceIntegrator::ComputeDivergenceImpl<3>(const ParameterFunction &pk,
                                                          Vector &R) const;
template void
NEML2StressDivergenceIntegrator::AssembleGradEAImpl<2>(Vector &emat);
template void
NEML2StressDivergenceIntegrator::AssembleGradEAImpl<3>(Vector &emat);

template void
NEML2StressDivergenceIntegrator::AssembleGradDiagonalPAImpl<2>(Vector &diag) const;
template void
NEML2StressDivergenceIntegrator::AssembleGradDiagonalPAImpl<3>(Vector &diag) const;

} // namespace mfem
