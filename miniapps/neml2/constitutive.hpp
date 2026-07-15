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

#include "mfem.hpp"

#include <ATen/ATen.h>
#include "neml2/csrc/dispatchers/DispatchedModel.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mfem
{
using namespace future;

/// @brief Description of one NEML2 variable (input or output).
///
/// `name` is the declared name (e.g. `stress`, `stress~1`, `X1`); `base` is that
/// name with any `~k` history-lag suffix stripped (`stress~1` -> `stress`).
/// `base_shape` is NEML2's per-point base shape (Scalar -> `{}`, SR2 -> `{6}`)
/// and `vdim` is its flattened size (`prod(base_shape)`, 1 for a Scalar).
struct VarSpec
{
   std::string name;
   std::string base;
   std::vector<int64_t> base_shape;
   int vdim = 0;
   bool is_scalar = false;
};

/// @brief Per-quadrature-point history store for one mesh level.
///
/// Small-strain plasticity is history-dependent: a NEML2 model consumes the
/// converged state of the previous load step (`stress~1`, `X1~1`,
/// `equivalent_plastic_strain~1`, the old strain/time, ...) and produces the new
/// state. This manager owns, for every stateful base variable, an `_old` (the
/// lagged value fed into the model this step) and a `_cur` (the value produced /
/// set this step) `ParameterFunction`, laid out per quadrature point exactly like
/// the integrator's strain/stress storage (same mesh + integration rule, so the
/// flattened `(nqp, vdim)` batch aligns for the name-keyed NEML2 call).
///
/// One instance lives per multigrid level (each on that level's mesh/IR) and
/// persists across the preconditioner factory's per-Newton rebuilds, so coarse
/// levels evolve their own history. `Reset()` seeds step 0 (all zero);
/// `Advance()` commits a converged step (`_old <- _cur`).
class MaterialStateManager
{
 public:
   /// Allocate `_old`/`_cur` storage for each variable in \p state_vars on the
   /// quadrature points of \p fes (using the same integration rule the
   /// NEML2StressDivergenceIntegrator picks for \p fes).
   MaterialStateManager(const FiniteElementSpace &fes,
                        const std::vector<VarSpec> &state_vars);

   /// Zero all `_old` and `_cur` (the step-0 seed).
   void Reset();

   /// Commit a converged step: copy `_cur` into `_old` for every variable.
   void Advance();

   /// Lagged (old-step) value fed into the model this step.
   ParameterFunction &Old(const std::string &base);
   /// Current value produced/set this step (staged for the next Advance()).
   ParameterFunction &Cur(const std::string &base);

   /// Whether \p base is a managed stateful variable.
   bool Has(const std::string &base) const;
   /// Whether this level carries any history at all (false for a stateless model).
   bool Empty() const { return _order.empty(); }

 private:
   std::vector<std::string> _order;
   std::map<std::string, std::unique_ptr<UniformParameterSpace>> _spaces;
   std::map<std::string, std::unique_ptr<ParameterFunction>> _old;
   std::map<std::string, std::unique_ptr<ParameterFunction>> _cur;
};

/// @brief Model-agnostic constitutive wrapper for NEML2 models (v3 cpp-aoti).
///
/// Wraps a `neml2::aoti::DispatchedModel` -- an offline-compiled (AOT-Inductor)
/// NEML2 model dispatched to one device by a scheduler. Evaluation goes through
/// the v3 name-keyed `forward` / `jvp` / `jacobian` surface over raw
/// `at::Tensor`s (see neml2/csrc/aoti/Model.h). Strain/stress are exchanged as
/// symmetric second-order tensors in NEML2's Mandel packing
/// `[xx, yy, zz, sqrt(2) yz, sqrt(2) xz, sqrt(2) xy]`.
///
/// The wrapper is model-agnostic: it takes the (plain, v3) names of the single
/// FE-driven strain input, the single stress output, and the time input, and
/// treats *every other* declared variable as generic stateful history. History
/// inputs are those whose name carries a `~k` lag suffix; their old values come
/// from a MaterialStateManager and their new values (the model outputs, plus the
/// externally supplied strain/time) are staged back into it. This lets one
/// compiled miniapp drive elasticity, J2, Chaboche, ... without recompilation.
class ConstitutiveModel
{
 public:
   /// @param cmodel     the loaded NEML2 dispatched model
   /// @param strain_var name of the FE-driven strain input (plain v3 name)
   /// @param stress_var name of the stress output (plain v3 name)
   /// @param time_var   name of the time input (plain v3 name)
   ConstitutiveModel(std::shared_ptr<neml2::aoti::DispatchedModel> cmodel,
                     const std::string &strain_var,
                     const std::string &stress_var,
                     const std::string &time_var);

   /**
   * @brief Stress update (residual path): stress = model(strain, time, old state)
   *
   * Evaluates the NEML2 `forward` and stages every model output into
   * `state.Cur(...)` so a converged step can be committed with Advance().
   *
   * @param strain Input strain (Mandel SR2 per quadrature point)
   * @param stress Output stress (Mandel SR2 per quadrature point)
   * @param time   current time
   * @param state  per-qp history store (old values read, new values staged)
   */
   void Mult(ParameterFunction &strain, ParameterFunction &stress, real_t time,
             MaterialStateManager &state) const;

   /**
   * @brief Consistent material tangent d(stress)/d(strain) at fixed old state.
   *
   * Evaluates the NEML2 `jacobian` once and returns the tangent in two forms
   * sharing that single solve: \p mandel_tangent is the raw SR2->SR2 Mandel 6x6
   * block `M[a,b] = d(stress_a)/d(strain_b)` (used for the matrix-free operator
   * action, see ApplyStoredTangent) and \p full_tangent is that block expanded to
   * a full 3x3x3x3 stiffness `C_ijkl` (used by the assembled paths: PA diagonal,
   * element assembly, coarse hypre matrix). Both are batched over quadrature
   * points, or batch-independent (`(6,6)` / `(3,3,3,3)`) when the block is (e.g.
   * linear elasticity). Also stages the model outputs into `state.Cur(...)` so
   * coarse levels capture their current state while the preconditioner is
   * assembled.
   *
   * Caching this consistent tangent per linearization is what makes the
   * matrix-free path affordable for implicit (plastic) models: each FE-level
   * operator apply would otherwise re-run the local return-map Newton via `jvp`.
   *
   * @param strain        Input strain (linearization point)
   * @param mandel_tangent Output Mandel 6x6 block, shape `(*B,6,6)` or `(6,6)`
   * @param full_tangent  Output `C_ijkl`, shape `(*B,3,3,3,3)` or `(3,3,3,3)`
   * @param time          current time
   * @param state         per-qp history store (old values read, new staged)
   */
   void Tangent(ParameterFunction &strain, at::Tensor &mandel_tangent,
                at::Tensor &full_tangent, real_t time,
                MaterialStateManager &state) const;

   /**
   * @brief Apply a cached material tangent to a strain increment, matrix-free.
   *
   * Computes d(stress) = M : d(strain) as a per-quadrature-point Mandel matvec
   * using the \p mandel_tangent produced by a prior Tangent() call at the current
   * linearization point. This is the exact consistent-tangent action MFEM's
   * partial assembly needs, without re-evaluating the constitutive model (the key
   * difference from a `jvp`, which would re-run the local solve on every apply).
   *
   * @param mandel_tangent Cached Mandel 6x6 block from Tangent()
   * @param dstrain        Input strain increment (direction, Mandel SR2)
   * @param dstress        Output stress increment (Mandel SR2)
   */
   void ApplyStoredTangent(const at::Tensor &mandel_tangent,
                           ParameterFunction &dstrain,
                           ParameterFunction &dstress) const;

   /// Stateful base variables this model needs history storage for (one per
   /// distinct history-input base name). A MaterialStateManager is built from
   /// this list. Empty for a stateless model (e.g. linear elasticity).
   const std::vector<VarSpec> &StateVars() const { return _state_vars; }

   /// Name of the FE-driven strain input.
   const std::string &StrainName() const { return _strain.name; }
   /// Name of the stress output.
   const std::string &StressName() const { return _stress.name; }
   /// Name of the time input (empty if the model declares no time input).
   const std::string &TimeName() const { return _time.name; }
   /// Whether the model declares the time variable as an input.
   bool HasTime() const { return _has_time; }
   /// Whether the model carries any history (needs per-step state advancement).
   bool HasState() const { return !_state_vars.empty(); }

   /// Base name of \p name with any trailing `~k` history-lag suffix removed.
   static std::string BaseName(const std::string &name);
   /// Whether \p name carries a `~k` history-lag suffix.
   static bool IsHistory(const std::string &name);

 private:
   std::map<std::string, at::Tensor> MakeInputs(ParameterFunction &strain,
                                                real_t time,
                                                MaterialStateManager &state) const;

   /// Stage every model output into `state.Cur(base)` (for the managed bases).
   void CaptureOutputs(const std::map<std::string, at::Tensor> &outputs,
                       MaterialStateManager &state) const;

   /// Stage the FE-driven strain input plus every model output into
   /// `state.Cur(...)` (the full set of current values a converged step commits).
   void CaptureState(ParameterFunction &strain,
                     const std::map<std::string, at::Tensor> &outputs,
                     MaterialStateManager &state) const;

   at::TensorOptions Options() const;

   /// The NEML2 constitutive model being wrapped
   std::shared_ptr<neml2::aoti::DispatchedModel> _cmodel;

   /// The FE-driven strain input, the stress output, and the time input.
   VarSpec _strain;
   VarSpec _stress;
   VarSpec _time;
   bool _has_time;

   /// Every input carrying a `~k` lag suffix (fed from `state.Old(base)`).
   std::vector<VarSpec> _history_inputs;
   /// Every model output (staged into `state.Cur(base)` when managed).
   std::vector<VarSpec> _outputs;
   /// One entry per distinct history-input base name -- the variables a
   /// MaterialStateManager stores for this model.
   std::vector<VarSpec> _state_vars;
};

} // namespace mfem
