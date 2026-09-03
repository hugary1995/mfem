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

#include "constitutive.hpp"

#include <ATen/ATen.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
#include <vector>

namespace mfem
{

// Wrap MFEM-owned quadrature data as a NEML2-shaped at::Tensor without copying.
// The parameter function stores `vdim` values per quadrature point contiguously,
// so the batched tensor is (num_qp, vdim) -- the canonical `(*B, *base_shape)`
// layout NEML2 expects for a per-point SR2 (base shape {6}); Scalars ({}) are
// reshaped down to (*B,) by the caller (see to_base_shape).
at::Tensor ConstitutiveModel::Wrap(const at::TensorOptions &options,
                                   ParameterFunction &pf)
{
   // Pointer to host/device data (MFEM owns it)
   real_t *data = pf.ReadWrite();

   const auto &q_space = pf.GetParameterSpace();
   const int64_t vdim = q_space.GetVDim();
   const int64_t nqp = q_space.GetTrueVSize() / vdim;

   // torch::from_blob wraps the data in place (no copy). `options` must name the
   // memory space the data actually lives in (see ConstitutiveModel::Options).
   return at::from_blob(data, {nqp, vdim}, options);
}

// Copy a NEML2 output tensor back into MFEM-owned quadrature storage. The tensor
// may be shaped (*B,), (*B,6), ... -- only its total element count matters, which
// must equal the parameter function's T-vector size.
static void neml2_to_mfem_tensor(const at::Tensor &neml2_tensor,
                                 ParameterFunction &pf)
{
   const auto &q_space = pf.GetParameterSpace();
   const at::Tensor contig = neml2_tensor.contiguous();
   // Alias the tensor data as an MFEM Vector, then assign (host/host or
   // device/device copy honoring the parameter function's memory space).
   Vector mfem_tensor(const_cast<real_t *>(contig.data_ptr<real_t>()),
                      q_space.GetTrueVSize());
   pf = mfem_tensor;
}

// Reshape a flat (nqp, vdim) quadrature tensor to NEML2's canonical
// (nqp, *base_shape): a no-op for SR2 ({6}), a squeeze of the trailing 1 for a
// Scalar ({}) -> (nqp,), an unflatten for an R2 ({3,3}) -> (nqp,3,3).
static at::Tensor to_base_shape(const at::Tensor &flat, const VarSpec &spec)
{
   std::vector<int64_t> shape;
   shape.push_back(flat.size(0));
   shape.insert(shape.end(), spec.base_shape.begin(), spec.base_shape.end());
   return flat.reshape(shape);
}

// Must match StressDivergenceIntegrator::SetUpQuadratureSpace exactly -- this is
// the rule the integrator picks for `fes` when constructed with a null rule.
// `IntRules` is a global cache, so this returns the very same rule object.
const IntegrationRule &NEML2IntRule(const FiniteElementSpace &fes)
{
   const auto &T = *fes.GetMesh()->GetTypicalElementTransformation();
   const int quad_order = 2 * T.OrderGrad(fes.GetTypicalFE());
   return IntRules.Get(T.GetGeometryType(), quad_order);
}

//
// MaterialStateManager
//

void MaterialStateManager::ApplyIC(const InitialCondition &ic,
                                   const VarSpec &var, ParameterFunction &pf)
{
   if (ic.kind == InitialCondition::Kind::Field)
   {
      MFEM_VERIFY(ic.field, "Field initial condition for '"
                               << var.base << "' has no field attached");
      MFEM_VERIFY(ic.field->Size() == pf.Size(),
                  "Field initial condition for '"
                     << var.base << "' has size " << ic.field->Size()
                     << ", expected " << pf.Size());
      pf.Vector::operator=(*ic.field);
      return;
   }

   // Everything else is a per-point pattern of `vdim` values repeated over the
   // quadrature points, so build one point's worth and broadcast it.
   std::vector<real_t> pattern(var.vdim, 0.0);
   if (ic.kind == InitialCondition::Kind::Constant)
   {
      std::fill(pattern.begin(), pattern.end(), ic.value);
   }
   else if (ic.kind == InitialCondition::Kind::Identity)
   {
      if (var.base_shape.empty())
      {
         pattern[0] = 1.0;
      }
      else if (var.base_shape == std::vector<int64_t>{3, 3})
      {
         // Row-major 3x3.
         pattern[0] = pattern[4] = pattern[8] = 1.0;
      }
      else if (var.base_shape == std::vector<int64_t>{6})
      {
         // Mandel SR2: the off-diagonal entries carry a sqrt(2) factor, which is
         // zero for the identity, so only the normal components are set.
         pattern[0] = pattern[1] = pattern[2] = 1.0;
      }
      else
      {
         MFEM_ABORT("No identity defined for variable '"
                    << var.base << "' with base shape of " << var.vdim
                    << " component(s)");
      }
   }

   const int n = pf.Size();
   real_t *data = pf.HostWrite();
   for (int i = 0; i < n; ++i)
   {
      data[i] = pattern[i % var.vdim];
   }
}

MaterialStateManager::MaterialStateManager(
   const FiniteElementSpace &fes, const std::vector<StateVarSpec> &state_vars,
   const std::vector<VarSpec> &fields)
{
   Mesh &mesh = *fes.GetMesh();
   const IntegrationRule &ir = NEML2IntRule(fes);

   for (const auto &s : state_vars)
   {
      const std::string &base = s.var.base;
      _order.push_back(base);
      _specs[base] = s;
      _spaces[base] =
         std::make_unique<UniformParameterSpace>(mesh, ir, s.var.vdim);
      auto &space = *_spaces[base];

      _cur[base] = std::make_unique<ParameterFunction>(space);
      _cur[base]->UseDevice(true);
      auto &ring = _hist[base];
      for (int k = 0; k < s.depth; ++k)
      {
         ring.push_back(std::make_unique<ParameterFunction>(space));
         ring.back()->UseDevice(true);
      }
   }

   for (const auto &f : fields)
   {
      _spaces[f.name] =
         std::make_unique<UniformParameterSpace>(mesh, ir, f.vdim);
      _fields[f.name] =
         std::make_unique<ParameterFunction>(*_spaces[f.name]);
      _fields[f.name]->UseDevice(true);
      *_fields[f.name] = 0.0;
   }

   Reset();
}

void MaterialStateManager::Reset()
{
   for (const auto &base : _order)
   {
      const auto &spec = _specs.at(base);
      ApplyIC(spec.ic, spec.var, *_cur[base]);
      for (auto &lagged : _hist[base])
      {
         ApplyIC(spec.ic, spec.var, *lagged);
      }
   }
}

void MaterialStateManager::Advance()
{
   for (const auto &base : _order)
   {
      auto &ring = _hist[base];
      // Rotate oldest-first so nothing is overwritten before it is read:
      // ~depth <- ~(depth-1), ..., ~2 <- ~1, ~1 <- cur.
      // ParameterFunction's copy-assignment is deleted (reference member); copy
      // the underlying data through the Vector base.
      for (int k = static_cast<int>(ring.size()) - 1; k > 0; --k)
      {
         ring[k]->Vector::operator=(*ring[k - 1]);
      }
      ring[0]->Vector::operator=(*_cur[base]);
   }
}

ParameterFunction &MaterialStateManager::Old(const std::string &base, int lag)
{
   auto it = _hist.find(base);
   MFEM_VERIFY(it != _hist.end(),
               "MaterialStateManager has no state variable '" << base << "'");
   MFEM_VERIFY(lag >= 1 && lag <= static_cast<int>(it->second.size()),
               "MaterialStateManager stores " << it->second.size()
                                              << " lag(s) for '" << base
                                              << "', asked for ~" << lag);
   return *it->second[lag - 1];
}

ParameterFunction &MaterialStateManager::Cur(const std::string &base)
{
   auto it = _cur.find(base);
   MFEM_VERIFY(it != _cur.end(),
               "MaterialStateManager has no state variable '" << base << "'");
   return *it->second;
}

ParameterFunction &MaterialStateManager::Field(const std::string &name)
{
   auto it = _fields.find(name);
   MFEM_VERIFY(it != _fields.end(),
               "MaterialStateManager has no prescribed field '" << name << "'");
   return *it->second;
}

bool MaterialStateManager::Has(const std::string &base) const
{
   return _cur.find(base) != _cur.end();
}

bool MaterialStateManager::HasField(const std::string &name) const
{
   return _fields.find(name) != _fields.end();
}

//
// ConstitutiveModel
//

int ConstitutiveModel::HistoryLag(const std::string &name)
{
   const auto pos = name.rfind('~');
   if (pos == std::string::npos || pos + 1 >= name.size())
   {
      return 0;
   }
   for (size_t i = pos + 1; i < name.size(); ++i)
   {
      if (!std::isdigit(static_cast<unsigned char>(name[i])))
      {
         return 0;
      }
   }
   return std::stoi(name.substr(pos + 1));
}

bool ConstitutiveModel::IsHistory(const std::string &name)
{
   return HistoryLag(name) > 0;
}

std::string ConstitutiveModel::BaseName(const std::string &name)
{
   return IsHistory(name) ? name.substr(0, name.rfind('~')) : name;
}

static VarSpec make_var_spec(const std::string &name,
                             const std::vector<int64_t> &base_shape)
{
   VarSpec s;
   s.name = name;
   s.base = ConstitutiveModel::BaseName(name);
   s.lag = ConstitutiveModel::HistoryLag(name);
   s.base_shape = base_shape;
   int vdim = 1;
   for (const auto d : base_shape)
   {
      vdim *= static_cast<int>(d);
   }
   s.vdim = vdim;
   s.is_scalar = base_shape.empty();
   return s;
}

// Expand a symmetric fourth-order tensor from NEML2's Mandel SSR4 6x6 packing
// into a full 3x3x3x3 stiffness `C_ijkl` (row-major, optionally batched).
//
// NEML2 SR2 packs a symmetric 2-tensor as [xx, yy, zz, sqrt2 yz, sqrt2 xz,
// sqrt2 xy]; the SSR4 6x6 M relates to the full C via
//   M[a,b] = mu_a mu_b C_(ij)(kl),  mu = 1 for a<3, sqrt(2) otherwise,
// so the inverse is C_ijkl = M[a,b] / (mu_a mu_b) with a = mandel(i,j),
// b = mandel(k,l). Because mandel(i,j) is symmetric, the result automatically
// carries both minor symmetries (C_ijkl = C_jikl = C_ijlk). This is the C++
// replacement for the retired `neml2::R4(neml2::SSR4(...))` conversion.
at::Tensor ConstitutiveModel::MandelToFullStiffness(const at::Tensor &M)
{
   using namespace at::indexing;
   static const int mandel[3][3] = {{0, 5, 4}, {5, 1, 3}, {4, 3, 2}};
   const real_t s2 = std::sqrt(real_t(2));
   const real_t mu[6] = {1, 1, 1, s2, s2, s2};

   // Output shape: replace the trailing (6,6) with (3,3,3,3).
   std::vector<int64_t> shape(M.sizes().begin(), M.sizes().end() - 2);
   shape.insert(shape.end(), {3, 3, 3, 3});
   at::Tensor C = at::empty(shape, M.options());

   for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
         for (int k = 0; k < 3; k++)
            for (int l = 0; l < 3; l++)
            {
               const int a = mandel[i][j], b = mandel[k][l];
               C.index_put_({Ellipsis, i, j, k, l},
                            M.index({Ellipsis, a, b}) / (mu[a] * mu[b]));
            }
   return C.contiguous();
}

ConstitutiveModel::ConstitutiveModel(
   std::shared_ptr<neml2::aoti::DispatchedModel> cmodel,
   const VariableBinding &binding)
    : _cmodel(cmodel), _mode(binding.mode), _has_time(false)
{
   const auto &in_names = _cmodel->input_names();
   const auto &in_shapes = _cmodel->input_base_shapes();
   const auto &out_names = _cmodel->output_names();
   const auto &out_shapes = _cmodel->output_base_shapes();

   const auto index_of = [&](const std::string &name) -> int
   {
      for (size_t i = 0; i < in_names.size(); ++i)
      {
         if (in_names[i] == name) { return static_cast<int>(i); }
      }
      return -1;
   };

   // Kinematics first, and in binding order -- Mult()/Tangent() index into this
   // list positionally, so it must match what the integrator hands over.
   const std::set<std::string> kinematic_names(binding.kinematics.begin(),
                                               binding.kinematics.end());
   const std::set<std::string> field_names(binding.fields.begin(),
                                           binding.fields.end());
   for (const auto &name : binding.kinematics)
   {
      const int i = index_of(name);
      MFEM_VERIFY(i >= 0, "NEML2 model declares no input named '"
                             << name << "' (bound as a kinematic input)");
      _kinematics.push_back(make_var_spec(in_names[i], in_shapes[i]));
   }

   // Classify the rest. Every declared input must land somewhere: an input the
   // driver was not told how to supply would otherwise be dropped here and
   // surface as an opaque failure inside the model call.
   for (size_t i = 0; i < in_names.size(); ++i)
   {
      const std::string &name = in_names[i];
      if (kinematic_names.count(name)) { continue; }

      if (!binding.time.empty() && name == binding.time)
      {
         _time = make_var_spec(name, in_shapes[i]);
         _has_time = true;
      }
      else if (IsHistory(name))
      {
         _history_inputs.push_back(make_var_spec(name, in_shapes[i]));
      }
      else if (field_names.count(name))
      {
         _field_vars.push_back(make_var_spec(name, in_shapes[i]));
      }
      else
      {
         MFEM_ABORT("NEML2 model declares input '"
                    << name
                    << "' which is not bound to a kinematic quantity, the time, "
                       "or a prescribed per-quadrature-point field, and carries "
                       "no ~k history suffix. Bind it explicitly (e.g. "
                       "--field "
                    << name << "=...).");
      }
   }

   // Classify outputs: every output is staged into the state store; one of them
   // is the stress the FE residual consumes.
   bool found_stress = false;
   for (size_t i = 0; i < out_names.size(); ++i)
   {
      _outputs.push_back(make_var_spec(out_names[i], out_shapes[i]));
      if (out_names[i] == binding.stress)
      {
         _stress = _outputs.back();
         found_stress = true;
      }
   }
   MFEM_VERIFY(found_stress, "NEML2 model declares no stress output named '"
                                << binding.stress << "'");

   // The stateful variables to store are the distinct history-input bases, each
   // kept to the deepest lag the model asks for. Their current values are
   // supplied either by a like-named model output (stress, Fp, ...) or
   // externally (the kinematic and time bases, set by the stepper).
   std::map<std::string, size_t> slot_of;
   for (const auto &h : _history_inputs)
   {
      auto it = slot_of.find(h.base);
      if (it == slot_of.end())
      {
         StateVarSpec s;
         s.var = h;
         s.var.name = h.base; // keyed by base name in the manager
         s.var.lag = 0;
         s.depth = h.lag;
         const auto ic = binding.initial_conditions.find(h.base);
         s.ic = (ic == binding.initial_conditions.end()) ? InitialCondition::Zero()
                                                         : ic->second;
         slot_of[h.base] = _state_vars.size();
         _state_vars.push_back(s);
      }
      else
      {
         auto &s = _state_vars[it->second];
         s.depth = std::max(s.depth, h.lag);
      }
   }
}

at::TensorOptions ConstitutiveModel::Options() const
{
   return at::TensorOptions().dtype(_cmodel->dtype()).device(_cmodel->device());
}

std::map<std::string, at::Tensor>
ConstitutiveModel::MakeInputs(const std::vector<ParameterFunction *> &kinematics,
                              real_t time, MaterialStateManager &state) const
{
   MFEM_VERIFY(kinematics.size() == _kinematics.size(),
               "ConstitutiveModel expects " << _kinematics.size()
                                            << " kinematic input(s), got "
                                            << kinematics.size());
   const auto options = Options();
   std::map<std::string, at::Tensor> inputs;

   int64_t nqp = 0;
   for (size_t i = 0; i < _kinematics.size(); ++i)
   {
      at::Tensor flat = ConstitutiveModel::Wrap(options, *kinematics[i]);
      nqp = flat.size(0);
      inputs[_kinematics[i].name] = to_base_shape(flat, _kinematics[i]);
   }

   if (_has_time)
   {
      // Scalar input: base shape {} -> shaped (*B,) = (nqp,)
      inputs[_time.name] = at::full({nqp}, time, options);
   }

   for (const auto &h : _history_inputs)
   {
      at::Tensor old =
         ConstitutiveModel::Wrap(options, state.Old(h.base, h.lag));
      inputs[h.name] = to_base_shape(old, h);
   }

   for (const auto &f : _field_vars)
   {
      at::Tensor field = ConstitutiveModel::Wrap(options, state.Field(f.name));
      inputs[f.name] = to_base_shape(field, f);
   }
   return inputs;
}

void ConstitutiveModel::CaptureState(
   const std::vector<ParameterFunction *> &kinematics,
   const std::map<std::string, at::Tensor> &outputs,
   MaterialStateManager &state) const
{
   // The kinematics are FE-driven inputs; their current values (to become the
   // lagged values next step) are not model outputs, so stage them here.
   for (size_t i = 0; i < _kinematics.size(); ++i)
   {
      if (state.Has(_kinematics[i].base))
      {
         // ParameterFunction's copy-assignment is deleted (reference member);
         // copy the underlying data through the Vector base.
         state.Cur(_kinematics[i].base).Vector::operator=(*kinematics[i]);
      }
   }
   for (const auto &o : _outputs)
   {
      if (state.Has(o.base))
      {
         neml2_to_mfem_tensor(outputs.at(o.name), state.Cur(o.base));
      }
   }
}

void ConstitutiveModel::Mult(const std::vector<ParameterFunction *> &kinematics,
                             ParameterFunction &stress, real_t time,
                             MaterialStateManager &state) const
{
   const auto inputs = MakeInputs(kinematics, time, state);
   const auto outputs = _cmodel->forward(inputs);
   neml2_to_mfem_tensor(outputs.at(_stress.name), stress);
   CaptureState(kinematics, outputs, state);
}

void ConstitutiveModel::Tangent(
   const std::vector<ParameterFunction *> &kinematics,
   std::vector<at::Tensor> &blocks, ParameterFunction &stress, real_t time,
   MaterialStateManager &state) const
{
   const auto inputs = MakeInputs(kinematics, time, state);
   auto [outputs, J] = _cmodel->jacobian(inputs);
   neml2_to_mfem_tensor(outputs.at(_stress.name), stress);
   // One d(stress)/d(kinematic) block per FE-driven input, batched over
   // quadrature points (or unbatched when batch-independent, e.g. linear
   // elasticity). Interpreting each block's packing is the integrator's job:
   // it depends on the kinematic measure, not on the model.
   const auto &rows = J.at(_stress.name);
   blocks.clear();
   blocks.reserve(_kinematics.size());
   for (const auto &kin : _kinematics)
   {
      blocks.push_back(rows.at(kin.name).contiguous());
   }
   // Stage current state so coarse levels capture history during preconditioner
   // assembly (their only evaluation path).
   CaptureState(kinematics, outputs, state);
}

void ConstitutiveModel::ApplyStoredTangent(const at::Tensor &block,
                                           ParameterFunction &dinput,
                                           ParameterFunction &dstress) const
{
   // d(stress)_a = M[a,b] d(input)_b, per quadrature point -- the exact
   // consistent tangent action, with no constitutive re-evaluation. `block` is
   // (*B, m, n) per qp or (m, n) when batch-independent; matmul broadcasts the
   // latter. The trailing base dimensions are flattened to a single index on
   // both sides, which is what the (nqp, vdim) storage already gives us.
   at::Tensor di = ConstitutiveModel::Wrap(Options(), dinput); // (nqp, n)
   const int64_t n = di.size(1);
   at::Tensor M = block.dim() > 2
                     ? block.reshape({block.size(0), -1, n})
                     : block.reshape({-1, n});
   at::Tensor ds = at::matmul(M, di.unsqueeze(-1)).squeeze(-1);
   neml2_to_mfem_tensor(ds, dstress);
}

} // namespace mfem
