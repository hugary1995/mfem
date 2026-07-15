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
// reshaped down to (*B,) by the caller (see ToBaseShape).
static at::Tensor mfem_to_neml2_tensor(const at::TensorOptions &options,
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
// Scalar ({}) -> (nqp,).
static at::Tensor to_base_shape(const at::Tensor &flat, const VarSpec &spec)
{
   std::vector<int64_t> shape;
   shape.push_back(flat.size(0));
   shape.insert(shape.end(), spec.base_shape.begin(), spec.base_shape.end());
   return flat.reshape(shape);
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
static at::Tensor mandel_ssr4_to_r4(const at::Tensor &M)
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

// Integration rule the NEML2StressDivergenceIntegrator picks for `fes` when
// constructed with a null rule -- must match
// StressDivergenceIntegrator::SetUpQuadratureSpace exactly so a
// MaterialStateManager's per-qp storage aligns with the integrator's strain /
// stress storage. `IntRules` is a global cache, so this returns the very same
// rule object the integrator resolves.
static const IntegrationRule &int_rule_for(const FiniteElementSpace &fes)
{
   const auto &T = *fes.GetMesh()->GetTypicalElementTransformation();
   const int quad_order = 2 * T.OrderGrad(fes.GetTypicalFE());
   return IntRules.Get(T.GetGeometryType(), quad_order);
}

//
// MaterialStateManager
//

MaterialStateManager::MaterialStateManager(const FiniteElementSpace &fes,
                                           const std::vector<VarSpec> &state_vars)
{
   Mesh &mesh = *fes.GetMesh();
   const IntegrationRule &ir = int_rule_for(fes);
   for (const auto &v : state_vars)
   {
      _order.push_back(v.base);
      _spaces[v.base] =
         std::make_unique<UniformParameterSpace>(mesh, ir, v.vdim);
      auto &space = *_spaces[v.base];
      _old[v.base] = std::make_unique<ParameterFunction>(space);
      _cur[v.base] = std::make_unique<ParameterFunction>(space);
      _old[v.base]->UseDevice(true);
      _cur[v.base]->UseDevice(true);
   }
   Reset();
}

void MaterialStateManager::Reset()
{
   for (const auto &base : _order)
   {
      *_old[base] = 0.0;
      *_cur[base] = 0.0;
   }
}

void MaterialStateManager::Advance()
{
   for (const auto &base : _order)
   {
      // ParameterFunction's copy-assignment is deleted (reference member); copy
      // the underlying data through the Vector base.
      _old[base]->Vector::operator=(*_cur[base]);
   }
}

ParameterFunction &MaterialStateManager::Old(const std::string &base)
{
   auto it = _old.find(base);
   MFEM_VERIFY(it != _old.end(),
               "MaterialStateManager has no state variable '" << base << "'");
   return *it->second;
}

ParameterFunction &MaterialStateManager::Cur(const std::string &base)
{
   auto it = _cur.find(base);
   MFEM_VERIFY(it != _cur.end(),
               "MaterialStateManager has no state variable '" << base << "'");
   return *it->second;
}

bool MaterialStateManager::Has(const std::string &base) const
{
   return _cur.find(base) != _cur.end();
}

//
// ConstitutiveModel
//

bool ConstitutiveModel::IsHistory(const std::string &name)
{
   const auto pos = name.rfind('~');
   if (pos == std::string::npos || pos + 1 >= name.size())
   {
      return false;
   }
   for (size_t i = pos + 1; i < name.size(); ++i)
   {
      if (!std::isdigit(static_cast<unsigned char>(name[i])))
      {
         return false;
      }
   }
   return true;
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

ConstitutiveModel::ConstitutiveModel(
   std::shared_ptr<neml2::aoti::DispatchedModel> cmodel,
   const std::string &strain_var, const std::string &stress_var,
   const std::string &time_var)
    : _cmodel(cmodel), _has_time(false)
{
   const auto &in_names = _cmodel->input_names();
   const auto &in_shapes = _cmodel->input_base_shapes();
   const auto &out_names = _cmodel->output_names();
   const auto &out_shapes = _cmodel->output_base_shapes();

   // Classify inputs: the one FE-driven strain, the (optional) time, and every
   // `~k`-lagged history variable. Everything else is ignored (a well-formed v3
   // solid-mechanics model has no other inputs).
   bool found_strain = false;
   for (size_t i = 0; i < in_names.size(); ++i)
   {
      if (in_names[i] == strain_var)
      {
         _strain = make_var_spec(in_names[i], in_shapes[i]);
         found_strain = true;
      }
      else if (in_names[i] == time_var)
      {
         _time = make_var_spec(in_names[i], in_shapes[i]);
         _has_time = true;
      }
      else if (IsHistory(in_names[i]))
      {
         _history_inputs.push_back(make_var_spec(in_names[i], in_shapes[i]));
      }
   }
   MFEM_VERIFY(found_strain, "NEML2 model declares no strain input named '"
                                << strain_var << "'");

   // Classify outputs: every output is staged into the state store; one of them
   // is the stress the FE residual consumes.
   bool found_stress = false;
   for (size_t i = 0; i < out_names.size(); ++i)
   {
      _outputs.push_back(make_var_spec(out_names[i], out_shapes[i]));
      if (out_names[i] == stress_var)
      {
         _stress = _outputs.back();
         found_stress = true;
      }
   }
   MFEM_VERIFY(found_stress, "NEML2 model declares no stress output named '"
                                << stress_var << "'");

   // The stateful variables to store are the distinct history-input bases. Their
   // current values are supplied either by a like-named model output (stress,
   // X1, ...) or externally (the strain and time bases, set by the stepper).
   std::set<std::string> seen;
   for (const auto &h : _history_inputs)
   {
      if (seen.insert(h.base).second)
      {
         VarSpec s = h;
         s.name = h.base; // keyed by base name in the manager
         _state_vars.push_back(s);
      }
   }
}

at::TensorOptions ConstitutiveModel::Options() const
{
   return at::TensorOptions().dtype(_cmodel->dtype()).device(_cmodel->device());
}

std::map<std::string, at::Tensor>
ConstitutiveModel::MakeInputs(ParameterFunction &strain, real_t time,
                              MaterialStateManager &state) const
{
   const auto options = Options();
   std::map<std::string, at::Tensor> inputs;

   at::Tensor strain_tensor = mfem_to_neml2_tensor(options, strain);
   const int64_t nqp = strain_tensor.size(0);
   inputs[_strain.name] = to_base_shape(strain_tensor, _strain);

   if (_has_time)
   {
      // Scalar input: base shape {} -> shaped (*B,) = (nqp,)
      inputs[_time.name] = at::full({nqp}, time, options);
   }

   for (const auto &h : _history_inputs)
   {
      at::Tensor old = mfem_to_neml2_tensor(options, state.Old(h.base));
      inputs[h.name] = to_base_shape(old, h);
   }
   return inputs;
}

void ConstitutiveModel::CaptureOutputs(
   const std::map<std::string, at::Tensor> &outputs,
   MaterialStateManager &state) const
{
   for (const auto &o : _outputs)
   {
      if (state.Has(o.base))
      {
         neml2_to_mfem_tensor(outputs.at(o.name), state.Cur(o.base));
      }
   }
}

void ConstitutiveModel::CaptureState(ParameterFunction &strain,
                                     const std::map<std::string, at::Tensor> &outputs,
                                     MaterialStateManager &state) const
{
   // The strain is the FE-driven input; its current value (to become the lagged
   // strain next step) is not a model output, so stage it from the FE side here.
   if (state.Has(_strain.base))
   {
      // ParameterFunction's copy-assignment is deleted (reference member); copy
      // the underlying data through the Vector base.
      state.Cur(_strain.base).Vector::operator=(strain);
   }
   CaptureOutputs(outputs, state);
}

void ConstitutiveModel::Mult(ParameterFunction &strain,
                             ParameterFunction &stress, real_t time,
                             MaterialStateManager &state) const
{
   const auto inputs = MakeInputs(strain, time, state);
   const auto outputs = _cmodel->forward(inputs);
   neml2_to_mfem_tensor(outputs.at(_stress.name), stress);
   CaptureState(strain, outputs, state);
}

void ConstitutiveModel::Tangent(ParameterFunction &strain,
                                at::Tensor &mandel_tangent,
                                at::Tensor &full_tangent, real_t time,
                                MaterialStateManager &state) const
{
   const auto inputs = MakeInputs(strain, time, state);
   auto [outputs, J] = _cmodel->jacobian(inputs);
   // J[stress][strain] is the SR2->SR2 Mandel 6x6 block (batched over qp, or
   // unbatched when the derivative is batch-independent). Keep it for the
   // matrix-free apply and expand it to the full C_ijkl for the assembled paths.
   mandel_tangent = J.at(_stress.name).at(_strain.name).contiguous();
   full_tangent = mandel_ssr4_to_r4(mandel_tangent);
   // Stage current state so coarse levels capture history during preconditioner
   // assembly (their only evaluation path).
   CaptureState(strain, outputs, state);
}

void ConstitutiveModel::ApplyStoredTangent(const at::Tensor &mandel_tangent,
                                           ParameterFunction &dstrain,
                                           ParameterFunction &dstress) const
{
   // d(stress)_a = M[a,b] d(strain)_b, per quadrature point. Both increments are
   // in Mandel packing, so this batched 6x6 matvec is the exact consistent
   // tangent action -- no constitutive re-evaluation. `mandel_tangent` is (*B,6,6)
   // (per qp) or (6,6) (batch-independent); matmul broadcasts the latter.
   at::Tensor de = mfem_to_neml2_tensor(Options(), dstrain); // (nqp, 6)
   at::Tensor ds = at::matmul(mandel_tangent, de.unsqueeze(-1)).squeeze(-1);
   neml2_to_mfem_tensor(ds, dstress);
}

} // namespace mfem
