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

#pragma once

#include "constitutive.hpp"
#include "mfem.hpp"

namespace mfem
{
class NEML2StressDivergenceIntegrator
    : public StressDivergenceIntegrator<NonlinearFormIntegrator>
{
 public:
   /**
   * @brief Construct a new Linear Momentum Balance object
   *
   * @param constit Shared, model-agnostic NEML2 constitutive wrapper (built once
   *                and shared across the top-level form and every multigrid level)
   * @param time Current simulation time
   * @param ir Integration rule for the quadrature
   */
   NEML2StressDivergenceIntegrator(
      std::shared_ptr<const ConstitutiveModel> constit, real_t time,
      const IntegrationRule *ir = nullptr);

   /// Set the current simulation time (updated once per load step).
   void SetTime(real_t time) { _t = time; }

   /// Set the (non-owning) per-quadrature-point history store this integrator
   /// evaluates against. Must be set before any residual/gradient evaluation.
   /// The converged strain and stress are staged into it automatically during
   /// residual/gradient evaluation (see ConstitutiveModel::CaptureState).
   void SetState(MaterialStateManager *state) { _state = state; }

   /// @name Constitutive timing (process-wide, across all integrator instances)
   ///
   /// Isolate the NEML2 constitutive cost -- the return-map `forward` (residual)
   /// and `jacobian` (tangent) solves -- from the surrounding linear algebra, so
   /// a solver comparison is not confounded by the (solver-independent) plasticity
   /// cost. Accumulated in static members so every level and every preconditioner
   /// rebuild contributes. When profiling is on, a device sync brackets each timed
   /// region for accurate GPU numbers (this serializes, so leave it off for
   /// production timing of the total). The matrix-free tangent apply (`M:dstrain`)
   /// is pure linear algebra and is intentionally not counted here.
   ///@{
   static void SetProfiling(bool profile) { s_profile = profile; }
   static void ResetConstitutiveTimers() { s_residual_time = s_tangent_time = 0.0; }
   /// Wall time (s) spent in NEML2 residual (`forward`) evaluations.
   static real_t ResidualConstitutiveTime() { return s_residual_time; }
   /// Wall time (s) spent in NEML2 tangent (`jacobian`) evaluations.
   static real_t TangentConstitutiveTime() { return s_tangent_time; }
   ///@}

   using StressDivergenceIntegrator<NonlinearFormIntegrator>::AssemblePA;
   void AssemblePA(const FiniteElementSpace &fes) override;

   /**
   * @brief Perform weak form evaluation
   *
   * @param x Input displacement E-vector
   * @param y Output residual E-vector
   */
   void AddMultPA(const Vector &X, Vector &R) const override;

   void AssembleGradPA(const Vector &x, const FiniteElementSpace &fes) override;
   template <int vdim> void AssembleGradDiagonalPAImpl(Vector &emat) const;
   void AssembleGradDiagonalPA(Vector &diag) const override;

   template <int vdim> void AssembleGradEAImpl(Vector &emat);
   void AssembleGradEA(const Vector &x, const FiniteElementSpace &fes,
                       Vector &emat) override;

   /**
    * Perform action of gradient (Jacobian) upon input vector \p x and put into \p y
    */
   void AddMultGradPA(const Vector &x, Vector &y) const override;

   template <int vdim>
   void ComputeStrainImpl(const Vector &X, ParameterFunction &strain) const;

   template <int vdim>
   void ComputeRImpl(const ParameterFunction &stress, Vector &R) const;

 private:
   real_t _t;

   /// Non-owning per-quadrature-point history store for this integrator's level
   /// (owned by main, persistent across preconditioner rebuilds).
   MaterialStateManager *_state = nullptr;

   /// Constitutive timing accumulators (see the profiling API above).
   static bool s_profile;
   static real_t s_residual_time;
   static real_t s_tangent_time;

   /// The quadrature space for symmetric 2nd order tensors
   std::unique_ptr<UniformParameterSpace> _q_space_symr2;

   /// The strain storage (scratch: residual strain / gradient-action direction)
   mutable std::unique_ptr<ParameterFunction> _strain;
   /// The strain at the current linearization point (set by AssembleGradPA,
   /// consumed by the matrix-free jvp gradient action in AddMultGradPA)
   mutable std::unique_ptr<ParameterFunction> _strain_lin;
   /// The stress storage
   mutable std::unique_ptr<ParameterFunction> _stress;
   /// Whether we're ordering by nodes or by vdim
   std::optional<Ordering::Type> _ordering;

   /// Material tangent as a full fourth-order stiffness C_ijkl, shape
   /// (*B,3,3,3,3) or (3,3,3,3) when batch-independent. Obtained from the NEML2
   /// jacobian at the linearization strain; used by the assembled paths.
   std::optional<at::Tensor> _tangent;

   /// The same tangent as the raw SR2->SR2 Mandel 6x6 block, shape (*B,6,6) or
   /// (6,6). Cached once per linearization (AssembleGradPA) and contracted with
   /// the strain increment by the matrix-free gradient action (AddMultGradPA), so
   /// each operator apply is a cheap 6x6 matvec instead of re-running the NEML2
   /// constitutive solve.
   std::optional<at::Tensor> _mandel_tangent;

   /// The model-agnostic NEML2 constitutive wrapper (shared, read-only).
   std::shared_ptr<const ConstitutiveModel> _constit_op;

   void ComputeStrain(const Vector &X, ParameterFunction &strain) const;
   void ComputeR(const ParameterFunction &stress, Vector &R) const;
};

} // namespace mfem
