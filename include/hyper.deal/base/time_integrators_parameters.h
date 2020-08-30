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

#ifndef HYPERDEAL_FUNCTIONALITIES_TIME_INGEGRATORS_PARAMTERS
#define HYPERDEAL_FUNCTIONALITIES_TIME_INGEGRATORS_PARAMTERS

#include <hyper.deal/base/config.h>

#include <deal.II/base/parameter_handler.h>

#include <functional>

namespace hyperdeal
{
  /**
   * Parameters of the class LowStorageRungeKuttaIntegrator.
   */
  struct LowStorageRungeKuttaIntegratorParamters
  {
    /**
     * Register parameters.
     */
    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.add_parameter("RKType",
                        type,
                        "Type of the low-storage Runge-Kutta scheme.");
    }

    /*
     * Type of Runge-Kutta scheme. Default: RK45.
     */
    std::string type = "rk45";
  };
} // namespace hyperdeal

#endif
