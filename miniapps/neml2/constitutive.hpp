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

/// @brief How the driver supplies one of a NEML2 model's declared inputs.
enum class InputRole
{
   /// Computed by the FE integrator from the displacement field each evaluation
   /// (small strain, deformation gradient, deformation rate, vorticity, ...).
   Kinematic,
   /// The current load-step time.
   Time,
   /// A `~k`-lagged value read from a MaterialStateManager.
   History,
   /// A per-quadrature-point field prescribed once and held fixed over the run
   /// (crystal orientation from a grain map being the motivating case).
   Field
};

/// @brief The integration rule the NEML2 coupling uses on \p fes.
///
/// Every per-quadrature-point store in the coupling -- the integrator's kinematic
/// and stress scratch, a level's MaterialStateManager, a prescribed orientation
/// field -- must be laid out on the same points, or the flattened `(nqp, vdim)`
/// batch handed to NEML2 will not align. This is the single definition they all
/// resolve through.
const IntegrationRule &NEML2IntRule(const FiniteElementSpace &fes);

/// @brief Which kinematic measure the FE side computes and hands the model.
///
/// This selects both what the integrator computes from the displacement field and
/// how it interprets the model's stress output, since the two are a matched pair:
/// a small-strain model returns a Cauchy stress work-conjugate to `sym(grad u)`,
/// a total-Lagrangian model returns a PK2 stress work-conjugate to `E(F)`.
enum class KinematicMode
{
   /// `sym(grad u)` as a Mandel SR2 (6 components); the model's stress output is
   /// the Cauchy stress, used directly in the divergence.
   SmallStrain,
   /// `F = I + grad u` as a row-major R2 (9 components), gradients taken with
   /// respect to the reference configuration; the model's stress output is the
   /// PK2 stress `S`, converted to `P = F S` for the divergence.
   DeformationGradient
};

/// @brief Description of one NEML2 variable (input or output).
///
/// `name` is the declared name (e.g. `stress`, `stress~1`, `X1`); `base` is that
/// name with any `~k` history-lag suffix stripped (`stress~1` -> `stress`), and
/// `lag` is that k (0 when the name carries no suffix). `base_shape` is NEML2's
/// per-point base shape (Scalar -> `{}`, SR2 -> `{6}`, R2 -> `{3,3}`) and `vdim`
/// is its flattened size (`prod(base_shape)`, 1 for a Scalar).
struct VarSpec
{
   std::string name;
   std::string base;
   std::vector<int64_t> base_shape;
   int vdim = 0;
   int lag = 0;
   bool is_scalar = false;
};

/// @brief How a state variable (or prescribed field) is initialized at step 0.
///
/// Plasticity models routinely need a non-zero starting state -- an initial slip
/// resistance, a plastic deformation gradient of `I` rather than `0`, a grain
/// orientation sampled per quadrature point. Zero is the right default only for
/// strain-like quantities.
struct InitialCondition
{
   enum class Kind
   {
      Zero,     ///< All components zero (the default).
      Constant, ///< Every component set to `value`.
      Identity, ///< The identity tensor for this variable's base shape.
      Field     ///< Copied from `field`, a per-quadrature-point ParameterFunction.
   };

   Kind kind = Kind::Zero;
   real_t value = 0.0;
   /// Non-owning; must outlive the manager and match its parameter space.
   const ParameterFunction *field = nullptr;

   static InitialCondition Zero() { return {}; }
   static InitialCondition Constant(real_t v)
   {
      return {Kind::Constant, v, nullptr};
   }
   static InitialCondition Identity() { return {Kind::Identity, 0.0, nullptr}; }
   static InitialCondition FromField(const ParameterFunction &f)
   {
      return {Kind::Field, 0.0, &f};
   }
};

/// @brief One stateful variable the manager must store.
struct StateVarSpec
{
   VarSpec var;
   /// Deepest `~k` lag the model asks for; the manager keeps this many old
   /// values. A model using only `x~1` needs depth 1; one whose predictor also
   /// reads `x~2` needs depth 2.
   int depth = 1;
   InitialCondition ic;
};

/// @brief Per-quadrature-point history store for one mesh level.
///
/// Plasticity is history-dependent: a NEML2 model consumes the converged state of
/// previous load steps (`stress~1`, `Fp~1`, `tauc~2`, the old strain/time, ...)
/// and produces the new state. This manager owns, for every stateful base
/// variable, a ring of `depth` lagged values (`~1` .. `~depth`) plus a `_cur`
/// (the value produced / set this step), laid out per quadrature point exactly
/// like the integrator's kinematic/stress storage (same mesh + integration rule,
/// so the flattened `(nqp, vdim)` batch aligns for the name-keyed NEML2 call).
///
/// It also owns any per-quadrature-point prescribed fields (grain orientation),
/// which share that layout but never advance.
///
/// One instance lives per multigrid level (each on that level's mesh/IR) and
/// persists across the preconditioner factory's per-Newton rebuilds, so coarse
/// levels evolve their own history. `Reset()` seeds step 0 from each variable's
/// InitialCondition; `Advance()` commits a converged step, rotating the ring.
class MaterialStateManager
{
 public:
   /// Allocate storage for each variable in \p state_vars, and for each
   /// prescribed field in \p fields, on the quadrature points of \p fes (using
   /// the same integration rule the NEML2StressDivergenceIntegrator picks).
   MaterialStateManager(const FiniteElementSpace &fes,
                        const std::vector<StateVarSpec> &state_vars,
                        const std::vector<VarSpec> &fields = {});

   /// Seed step 0: apply each variable's initial condition to `_cur` and to
   /// every lag in its ring.
   void Reset();

   /// Commit a converged step: rotate each ring (`~k <- ~(k-1)`, `~1 <- _cur`).
   void Advance();

   /// Lagged value fed into the model this step; \p lag is 1-based (`x~1` is
   /// lag 1).
   ParameterFunction &Old(const std::string &base, int lag = 1);
   /// Current value produced/set this step (staged for the next Advance()).
   ParameterFunction &Cur(const std::string &base);
   /// A per-quadrature-point prescribed field, by declared name.
   ParameterFunction &Field(const std::string &name);

   /// Whether \p base is a managed stateful variable.
   bool Has(const std::string &base) const;
   /// Whether \p name is a managed prescribed field.
   bool HasField(const std::string &name) const;
   /// Whether this level carries any history at all (false for a stateless model).
   bool Empty() const { return _order.empty(); }

 private:
   /// Write \p ic into \p pf, given the variable's base shape.
   static void ApplyIC(const InitialCondition &ic, const VarSpec &var,
                       ParameterFunction &pf);

   std::vector<std::string> _order;
   std::map<std::string, StateVarSpec> _specs;
   std::map<std::string, std::unique_ptr<UniformParameterSpace>> _spaces;
   /// `_hist[base][k-1]` is the `~k` value.
   std::map<std::string, std::vector<std::unique_ptr<ParameterFunction>>> _hist;
   std::map<std::string, std::unique_ptr<ParameterFunction>> _cur;
   std::map<std::string, std::unique_ptr<ParameterFunction>> _fields;
};

/// @brief Names binding the driver's roles to a model's declared variables.
///
/// The FE side can produce a handful of kinematic quantities; which of them a
/// given NEML2 model wants, and what it calls them, is the model's business. The
/// driver therefore passes an explicit mapping rather than assuming names, so one
/// compiled miniapp drives small-strain, rate-form and multiplicative models
/// without recompilation.
struct VariableBinding
{
   /// Which kinematic measure the FE side computes; also fixes how the stress
   /// output is interpreted.
   KinematicMode mode = KinematicMode::SmallStrain;
   /// FE-driven inputs, in the order the integrator will supply them.
   std::vector<std::string> kinematics;
   /// The stress output the FE residual consumes.
   std::string stress;
   /// The time input (empty if the model declares none).
   std::string time;
   /// Inputs to treat as per-quadrature-point prescribed fields. Any declared
   /// input that is neither kinematic, time, nor `~k`-lagged must appear here;
   /// otherwise the model would be called with a missing input.
   std::vector<std::string> fields;
   /// Initial conditions by base name; absent entries default to zero.
   std::map<std::string, InitialCondition> initial_conditions;
};

/// @brief Model-agnostic constitutive wrapper for NEML2 models (v3 cpp-aoti).
///
/// Wraps a `neml2::aoti::DispatchedModel` -- an offline-compiled (AOT-Inductor)
/// NEML2 model dispatched to one device by a scheduler. Evaluation goes through
/// the v3 name-keyed `forward` / `jvp` / `jacobian` surface over raw
/// `at::Tensor`s (see neml2/csrc/aoti/Model.h). Symmetric second-order tensors
/// are exchanged in NEML2's Mandel packing
/// `[xx, yy, zz, sqrt(2) yz, sqrt(2) xz, sqrt(2) xy]`; full second-order tensors
/// (a deformation gradient) as row-major 3x3.
///
/// The wrapper is model-agnostic: a VariableBinding says which declared inputs
/// the FE side drives, which is the stress, which is time, and which are
/// prescribed per-point fields; everything carrying a `~k` lag suffix is generic
/// history. Every declared input must fall into exactly one of those buckets --
/// the constructor rejects a model with an input it was not told how to supply,
/// rather than silently dropping it.
class ConstitutiveModel
{
 public:
   /// @param cmodel  the loaded NEML2 dispatched model
   /// @param binding role-to-name mapping for this model
   ConstitutiveModel(std::shared_ptr<neml2::aoti::DispatchedModel> cmodel,
                     const VariableBinding &binding);

   /**
   * @brief Stress update (residual path): stress = model(kinematics, time, old state)
   *
   * Evaluates the NEML2 `forward` and stages every model output into
   * `state.Cur(...)` so a converged step can be committed with Advance().
   *
   * @param kinematics One ParameterFunction per KinematicVars() entry, in order
   * @param stress     Output stress (Mandel SR2 per quadrature point)
   * @param time       current time
   * @param state      per-qp history store (old values read, new values staged)
   */
   void Mult(const std::vector<ParameterFunction *> &kinematics,
             ParameterFunction &stress, real_t time,
             MaterialStateManager &state) const;

   /**
   * @brief Consistent material tangent d(stress)/d(kinematic) at fixed old state.
   *
   * Evaluates the NEML2 `jacobian` once and returns the derivative of the stress
   * with respect to each kinematic input, in declaration order:
   * `blocks[i][a,b] = d(stress_a)/d(kinematics_i_b)`, batched over quadrature
   * points or batch-independent when the block is (e.g. linear elasticity).
   *
   * Caching this consistent tangent per linearization is what makes the
   * matrix-free path affordable for implicit (plastic) models: each FE-level
   * operator apply would otherwise re-run the local return-map Newton via `jvp`.
   *
   * @param kinematics Linearization point, one per KinematicVars() entry
   * @param blocks     Output, one raw Jacobian block per kinematic input
   * @param stress     Output stress at the linearization point. Finite-deformation
   *                   kinematics need it to form the geometric stiffness, and it
   *                   comes free with the jacobian call.
   * @param time       current time
   * @param state      per-qp history store (old values read, new staged)
   */
   void Tangent(const std::vector<ParameterFunction *> &kinematics,
                std::vector<at::Tensor> &blocks, ParameterFunction &stress,
                real_t time, MaterialStateManager &state) const;

   /**
   * @brief Apply a cached tangent block to a kinematic increment, matrix-free.
   *
   * Computes d(stress) = M : d(kinematic) as a per-quadrature-point matvec using
   * a block produced by a prior Tangent() call at the current linearization
   * point. This is the exact consistent-tangent action MFEM's partial assembly
   * needs, without re-evaluating the constitutive model (the key difference from
   * a `jvp`, which would re-run the local solve on every apply).
   *
   * @param block  Cached Jacobian block from Tangent()
   * @param dinput Input increment (direction)
   * @param dstress Output stress increment (Mandel SR2)
   */
   void ApplyStoredTangent(const at::Tensor &block, ParameterFunction &dinput,
                           ParameterFunction &dstress) const;

   /// Expand a Mandel SSR4 6x6 block into a full 3x3x3x3 stiffness `C_ijkl`.
   /// Only valid for a small-strain (SR2 -> SR2) block.
   static at::Tensor MandelToFullStiffness(const at::Tensor &mandel);

   /// Stateful variables this model needs history storage for, with the depth
   /// and initial condition resolved. A MaterialStateManager is built from this
   /// list. Empty for a stateless model (e.g. linear elasticity).
   const std::vector<StateVarSpec> &StateVars() const { return _state_vars; }
   /// Per-quadrature-point prescribed fields this model reads.
   const std::vector<VarSpec> &FieldVars() const { return _field_vars; }
   /// FE-driven inputs, in the order Mult()/Tangent() expect them.
   const std::vector<VarSpec> &KinematicVars() const { return _kinematics; }

   /// Name of the stress output.
   const std::string &StressName() const { return _stress.name; }
   /// Name of the time input (empty when the model declares none).
   const std::string &TimeName() const { return _time.name; }
   /// The kinematic measure this model was bound to.
   KinematicMode Mode() const { return _mode; }
   /// Whether the model declares the time variable as an input.
   bool HasTime() const { return _has_time; }
   /// Whether the model carries any history (needs per-step state advancement).
   bool HasState() const { return !_state_vars.empty(); }

   /// Base name of \p name with any trailing `~k` history-lag suffix removed.
   static std::string BaseName(const std::string &name);
   /// Whether \p name carries a `~k` history-lag suffix.
   static bool IsHistory(const std::string &name);
   /// The k in a `~k` suffix, or 0 when \p name carries none.
   static int HistoryLag(const std::string &name);

   /// Tensor options (dtype and device) of the dispatched model, for wrapping
   /// MFEM quadrature storage as an at::Tensor in the same memory space.
   at::TensorOptions Options() const;

   /// Alias MFEM-owned quadrature data as a `(num_qp, vdim)` at::Tensor without
   /// copying. \p options must name the memory space the data actually lives in.
   static at::Tensor Wrap(const at::TensorOptions &options,
                          ParameterFunction &pf);

 private:
   std::map<std::string, at::Tensor>
   MakeInputs(const std::vector<ParameterFunction *> &kinematics, real_t time,
              MaterialStateManager &state) const;

   /// Stage the FE-driven kinematics plus every model output into
   /// `state.Cur(...)` (the full set of current values a converged step commits).
   void CaptureState(const std::vector<ParameterFunction *> &kinematics,
                     const std::map<std::string, at::Tensor> &outputs,
                     MaterialStateManager &state) const;

   /// The NEML2 constitutive model being wrapped
   std::shared_ptr<neml2::aoti::DispatchedModel> _cmodel;

   KinematicMode _mode;

   /// FE-driven inputs, in binding order.
   std::vector<VarSpec> _kinematics;
   /// The stress output and the time input.
   VarSpec _stress;
   VarSpec _time;
   bool _has_time;

   /// Every input carrying a `~k` lag suffix (fed from `state.Old(base, k)`).
   std::vector<VarSpec> _history_inputs;
   /// Every per-quadrature-point prescribed input (fed from `state.Field(name)`).
   std::vector<VarSpec> _field_vars;
   /// Every model output (staged into `state.Cur(base)` when managed).
   std::vector<VarSpec> _outputs;
   /// One entry per distinct history-input base name -- the variables a
   /// MaterialStateManager stores for this model.
   std::vector<StateVarSpec> _state_vars;
};

} // namespace mfem
