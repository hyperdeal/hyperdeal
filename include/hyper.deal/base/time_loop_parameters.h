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

#ifndef HYPERDEAL_FUNCTIONALITIES_TIME_LOOP_PARAMTERS
#define HYPERDEAL_FUNCTIONALITIES_TIME_LOOP_PARAMTERS

#include <hyper.deal/base/config.h>

#include <deal.II/base/parameter_handler.h>

#include <functional>

namespace hyperdeal
{
  template <typename Number>
  struct TimeLoopParamters
  {
    void
    add_parameters(dealii::ParameterHandler &prm)
    {
      prm.add_parameter(
        "TimeStep",
        time_step,
        "Time-step size: we take the minimum of this quantity and the maximum time step obtained by the CFL condition.");
      prm.add_parameter("StartTime", start_time, "Start time.");
      prm.add_parameter("FinalTime", final_time, "Final time.");
      prm.add_parameter(
        "MaxTimeStepNumber",
        max_time_step_number,
        "Terminate simulation after a maximum number of iterations (useful for performance studies).");
    }

    Number       time_step            = 0.1;
    Number       start_time           = 0.0;
    Number       final_time           = 20.0;
    unsigned int max_time_step_number = 100000000;
  };
} // namespace hyperdeal

#endif