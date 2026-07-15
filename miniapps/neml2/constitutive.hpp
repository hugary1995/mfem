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

namespace mfem
{
using namespace future;

/// @brief Constitutive model wrapper for NEML2 models (v3 cpp-aoti route)
///
/// Wraps a `neml2::aoti::DispatchedModel` -- an offline-compiled (AOT-Inductor)
/// NEML2 model dispatched to one device by a scheduler. Evaluation goes through
/// the v3 name-keyed `forward` / `jvp` / `jacobian` surface over raw
/// `at::Tensor`s (see neml2/csrc/aoti/Model.h). Strain/stress are exchanged as
/// symmetric second-order tensors in NEML2's Mandel packing
/// `[xx, yy, zz, sqrt(2) yz, sqrt(2) xz, sqrt(2) xy]`.
class ConstitutiveModel
{
 public:
   ConstitutiveModel(std::shared_ptr<neml2::aoti::DispatchedModel> cmodel);

   /**
   * @brief Perform stress update (residual path): stress = model(strain, time)
   *
   * @param strain Input strain (Mandel SR2 per quadrature point)
   * @param stress Output stress (Mandel SR2 per quadrature point)
   * @param time current time
   */
   void Mult(ParameterFunction &strain, ParameterFunction &stress,
             real_t time) const;

   /**
   * @brief Compute the material tangent d(stress)/d(strain) as a full
   *        fourth-order tensor.
   *
   * Evaluates the NEML2 `jacobian` (the SR2->SR2 Mandel 6x6 block) and expands
   * it into a full 3x3x3x3 stiffness \p tangent laid out row-major as
   * `C_ijkl` (optionally batched over quadrature points). Used by the assembled
   * paths (element assembly, PA diagonal, coarse hypre matrix).
   *
   * @param strain Input strain (linearization point)
   * @param tangent Output tangent `C_ijkl`, shape `(*B,3,3,3,3)` or `(3,3,3,3)`
   *                when the block is batch-independent (e.g. linear elasticity)
   * @param time current time
   */
   void Tangent(ParameterFunction &strain, at::Tensor &tangent,
                real_t time) const;

   /**
   * @brief Apply the material tangent to a strain increment, matrix-free.
   *
   * Uses the NEML2 `jvp` (Jacobian-vector product) evaluated at the
   * linearization strain \p strain_lin: this is exactly the directional
   * derivative d(stress) = C(strain_lin) : d(strain) that MFEM's partial
   * assembly needs, without ever forming the full tangent.
   *
   * @param strain_lin Strain at the current linearization point
   * @param dstrain Input strain increment (direction)
   * @param dstress Output stress increment
   * @param time current time
   */
   void ApplyTangent(ParameterFunction &strain_lin, ParameterFunction &dstrain,
                     ParameterFunction &dstress, real_t time) const;

 private:
   std::map<std::string, at::Tensor> MakeInputs(ParameterFunction &strain,
                                                real_t time) const;

   at::TensorOptions Options() const;

   /// The NEML2 constitutive model being wrapped
   std::shared_ptr<neml2::aoti::DispatchedModel> _cmodel;

   /// Qualified NEML2 variable names
   const std::string _time_name;
   const std::string _strain_name;
   const std::string _stress_name;

   /// Whether the model declares the time variable as an input
   bool _has_time;
};

} // namespace mfem
