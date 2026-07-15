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
#include <cmath>
#include <vector>

namespace mfem
{

// Wrap MFEM-owned quadrature data as a NEML2-shaped at::Tensor without copying.
// The parameter function stores `vdim` values per quadrature point contiguously,
// so the batched tensor is (num_qp, vdim) -- the canonical `(*B, *base_shape)`
// layout NEML2 expects for a per-point SR2 (base shape {6}).
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

// Copy a NEML2 output tensor back into MFEM-owned quadrature storage.
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

// Expand a symmetric fourth-order tensor from NEML2's Mandel SSR4 6x6 packing
// into a full 3x3x3x3 stiffness `C_ijkl` (row-major, optionally batched).
//
// NEML2 SR2 packs a symmetric 2-tensor as [xx, yy, zz, sqrt2 yz, sqrt2 xz,
// sqrt2 xy]; the SSR4 6x6 M relates to the full C via
//   M[a,b] = mu_a mu_b C_(ij)(kl),  mu = 1 for a<3, sqrt(2) otherwise,
// so the inverse is C_ijkl = M[a,b] / (mu_a mu_b) with a = mandel(i,j),
// b = mandel(k,l). This is the C++ replacement for the retired
// `neml2::R4(neml2::SSR4(...))` conversion.
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

ConstitutiveModel::ConstitutiveModel(
   std::shared_ptr<neml2::aoti::DispatchedModel> cmodel)
   : _cmodel(cmodel), _time_name("forces/t"), _strain_name("forces/strain"),
     _stress_name("state/stress")
{
   const auto &names = _cmodel->input_names();
   _has_time = std::find(names.begin(), names.end(), _time_name) != names.end();
}

at::TensorOptions ConstitutiveModel::Options() const
{
   return at::TensorOptions().dtype(_cmodel->dtype()).device(_cmodel->device());
}

std::map<std::string, at::Tensor>
ConstitutiveModel::MakeInputs(ParameterFunction &strain, real_t time) const
{
   const auto options = Options();
   at::Tensor strain_tensor = mfem_to_neml2_tensor(options, strain);
   std::map<std::string, at::Tensor> inputs = {{_strain_name, strain_tensor}};
   if (_has_time)
   {
      // Scalar input: base shape {} -> shaped (*B,) = (num_qp,)
      inputs[_time_name] = at::full({strain_tensor.size(0)}, time, options);
   }
   return inputs;
}

void ConstitutiveModel::Mult(ParameterFunction &strain,
                             ParameterFunction &stress, real_t time) const
{
   const auto inputs = MakeInputs(strain, time);
   const auto outputs = _cmodel->forward(inputs);
   neml2_to_mfem_tensor(outputs.at(_stress_name), stress);
}

void ConstitutiveModel::Tangent(ParameterFunction &strain, at::Tensor &tangent,
                                real_t time) const
{
   const auto inputs = MakeInputs(strain, time);
   auto [outputs, J] = _cmodel->jacobian(inputs);
   // J[stress][strain] is the SR2->SR2 Mandel 6x6 block (batched over qp, or
   // unbatched when the derivative is batch-independent).
   tangent = mandel_ssr4_to_r4(J.at(_stress_name).at(_strain_name));
}

void ConstitutiveModel::ApplyTangent(ParameterFunction &strain_lin,
                                     ParameterFunction &dstrain,
                                     ParameterFunction &dstress,
                                     real_t time) const
{
   const auto inputs = MakeInputs(strain_lin, time);
   at::Tensor dstrain_tensor = mfem_to_neml2_tensor(Options(), dstrain);
   const std::map<std::string, at::Tensor> tangents = {
      {_strain_name, dstrain_tensor}};
   // jvp returns {outputs, jvp_outputs}; jvp_outputs[stress] is the directional
   // derivative d(stress) = C(strain_lin) : d(strain).
   auto [outputs, jvp_outputs] = _cmodel->jvp(inputs, tangents);
   neml2_to_mfem_tensor(jvp_outputs.at(_stress_name), dstress);
}

} // namespace mfem
