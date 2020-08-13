// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the hyper.deal authors
//
// This file is part of the hyper.deal library.
//
// The hyper.deal library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hyper.deal.
//
// ---------------------------------------------------------------------

#ifndef HYPERDEAL_NDIM_OPERATORS_ADVECTIONOPERATION_PARAMTERS
#define HYPERDEAL_NDIM_OPERATORS_ADVECTIONOPERATION_PARAMTERS

#include <hyper.deal/base/config.h>

#include <deal.II/base/parameter_handler.h>

#include <functional>

namespace hyperdeal
{
  namespace advection
  {
    struct AdvectionOperationParamters
    {
      void
      add_parameters(dealii::ParameterHandler &prm)
      {
        prm.add_parameter(
          "SkewFactor",
          factor_skew,
          "Factor to blend between conservative and convective implementation of DG.");
      }

      // skew factor: conservative (skew=0) and convective (skew=1)
      double factor_skew = 0.0;
    };
  } // namespace advection
} // namespace hyperdeal

#endif
