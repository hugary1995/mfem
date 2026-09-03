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
/// @brief Stress-divergence integrator driven by a NEML2 constitutive model.
///
/// Handles both small-strain and total-Lagrangian finite-deformation kinematics
/// (see KinematicMode) through one set of kernels. The unification comes from
/// writing both residuals in the same reference-configuration form,
///
///   R_(I,i) = \int T_iJ dphi_I/dX_J dV0,
///
/// with `T = sigma` under small strain (where the reference and current
/// configurations coincide) and `T = P = F S` under finite deformation, and both
/// tangents as the same non-symmetric fourth-order object
///
///   D_iJkL = dT_iJ / d(du_k/dX_L),
///
/// which is `C_ijkl` under small strain and `delta_ik S_JL + F_iM dS_MJ/dF_kL`
/// (geometric plus material) under finite deformation. Everything downstream of
/// those two quantities -- the divergence reduction, the matrix-free gradient
/// action, the PA diagonal, and element assembly -- is then kinematics-agnostic.
class NEML2StressDivergenceIntegrator
    : public StressDivergenceIntegrator<NonlinearFormIntegrator>
{
 public:
   /**
   * @brief Construct a new Linear Momentum Balance object
   *
   * @param constit Shared, model-agnostic NEML2 constitutive wrapper (built once
   *                and shared across the top-level form and every multigrid
   *                level). Its binding fixes the kinematic measure this
   *                integrator computes; see KinematicMode.
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
   /// The converged kinematics and model outputs are staged into it automatically
   /// during residual/gradient evaluation (see ConstitutiveModel::CaptureState).
   void SetState(MaterialStateManager *state) { _state = state; }

   /// @name Constitutive timing (process-wide, across all integrator instances)
   ///
   /// Isolate the NEML2 constitutive cost -- the return-map `forward` (residual)
   /// and `jacobian` (tangent) solves -- from the surrounding linear algebra, so
   /// a solver comparison is not confounded by the (solver-independent) plasticity
   /// cost. Accumulated in static members so every level and every preconditioner
   /// rebuild contributes. When profiling is on, a device sync brackets each timed
   /// region for accurate GPU numbers (this serializes, so leave it off for
   /// production timing of the total). The matrix-free tangent apply (`D:grad du`)
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

   /// Interpolate `sym(grad u)` to the quadrature points as a Mandel SR2.
   template <int vdim>
   void ComputeStrainImpl(const Vector &X, ParameterFunction &strain) const;

   /// Interpolate `grad u` (with respect to the reference configuration) to the
   /// quadrature points as a row-major 3x3, optionally adding the identity to
   /// form the deformation gradient `F`. Always 9 components: in 2D the
   /// out-of-plane row/column is zero (or the identity's 1) so NEML2, which is
   /// always 3D, sees a consistent plane-strain tensor.
   template <int vdim>
   void ComputeGradUImpl(const Vector &X, bool add_identity,
                         ParameterFunction &gradu) const;

   /// Reduce a quadrature-point stress `T_iJ` (row-major 3x3) against the
   /// reference-configuration shape function gradients into an E-vector residual.
   template <int vdim>
   void ComputeDivergenceImpl(const ParameterFunction &pk, Vector &R) const;

   /// Build `_pk` (the stress paired with the shape function gradients) from the
   /// model's stress output and, under finite deformation, the deformation
   /// gradient in \p kin. Public because it launches a device kernel, which nvcc
   /// will not accept from a non-public member.
   void ComputePK(const std::vector<std::unique_ptr<ParameterFunction>> &kin)
                                                                                const;

 private:
   real_t _t;
   KinematicMode _mode;

   /// Non-owning per-quadrature-point history store for this integrator's level
   /// (owned by main, persistent across preconditioner rebuilds).
   MaterialStateManager *_state = nullptr;

   /// Constitutive timing accumulators (see the profiling API above).
   static bool s_profile;
   static real_t s_residual_time;
   static real_t s_tangent_time;

   /// Quadrature spaces for the per-point tensors we exchange with NEML2: a
   /// Mandel SR2 (6), a full row-major 3x3 (9), and whichever of the two the
   /// active kinematic measure uses.
   std::unique_ptr<UniformParameterSpace> _q_space_symr2;
   std::unique_ptr<UniformParameterSpace> _q_space_r2;
   /// One space per ConstitutiveModel::KinematicVars() entry, sized by its vdim.
   std::vector<std::unique_ptr<UniformParameterSpace>> _kin_spaces;

   /// Kinematic input storage (scratch, residual path), one per
   /// ConstitutiveModel::KinematicVars() entry.
   mutable std::vector<std::unique_ptr<ParameterFunction>> _kin;
   /// The same at the current linearization point (set by AssembleGradPA).
   mutable std::vector<std::unique_ptr<ParameterFunction>> _kin_lin;
   /// Raw pointers into the two vectors above, in KinematicVars() order, for the
   /// ConstitutiveModel calls.
   mutable std::vector<ParameterFunction *> _kin_ptrs, _kin_lin_ptrs;

   /// The model's stress output (Mandel SR2): Cauchy under small strain, PK2
   /// under finite deformation.
   mutable std::unique_ptr<ParameterFunction> _stress;
   /// The stress paired with the reference-configuration shape function
   /// gradients in the divergence (`sigma` or `P = F S`), row-major 3x3.
   mutable std::unique_ptr<ParameterFunction> _pk;
   /// `grad du` at the quadrature points, the direction of the matrix-free
   /// gradient action, row-major 3x3.
   mutable std::unique_ptr<ParameterFunction> _gradu;
   /// Whether we're ordering by nodes or by vdim
   std::optional<Ordering::Type> _ordering;

   /// The tangent `D_iJkL = dT_iJ/d(du_k/dX_L)`, shape (*B,3,3,3,3) or
   /// (3,3,3,3) when batch-independent (linear elasticity). Built by
   /// AssembleGradPA from the NEML2 jacobian at the linearization point and
   /// consumed by every gradient path. Non-symmetric in general, so the kernels
   /// contract it in the unambiguous `dphiI[J] D(i,J,k,L) dphiJ[L]` order rather
   /// than relying on major symmetry.
   std::optional<at::Tensor> _tangent;

   /// The same tangent flattened to (*B,9,9) or (9,9), so the matrix-free
   /// gradient action is a per-quadrature-point matvec against `grad du`. Cached
   /// once per linearization so each operator apply skips the NEML2 solve.
   std::optional<at::Tensor> _flat_tangent;

   /// The model-agnostic NEML2 constitutive wrapper (shared, read-only).
   std::shared_ptr<const ConstitutiveModel> _constit_op;

   /// Fill \p kin (in KinematicVars() order) from the displacement E-vector.
   void ComputeKinematics(const Vector &X,
                          std::vector<std::unique_ptr<ParameterFunction>> &kin)
                                                                                const;
   void ComputeDivergence(const ParameterFunction &pk, Vector &R) const;
   void ComputeGradU(const Vector &X, bool add_identity,
                     ParameterFunction &gradu) const;
};

} // namespace mfem
