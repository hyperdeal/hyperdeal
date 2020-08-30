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

#include <hyper.deal/base/time_loop.h>
#include <hyper.deal/lac/sm_vector.h>

namespace hyperdeal
{
  template <typename Number, typename VectorType>
  void
  TimeLoop<Number, VectorType>::reinit(
    const TimeLoopParamters<Number> &parameters)
  {
    this->time_step            = parameters.time_step;
    this->start_time           = parameters.start_time;
    this->final_time           = parameters.final_time;
    this->max_time_step_number = parameters.max_time_step_number;
  }

  template <typename Number, typename VectorType>
  int
  TimeLoop<Number, VectorType>::loop(
    VectorType &solution,
    const std::function<void(
      VectorType &,
      const Number,
      const Number,
      const std::function<void(const VectorType &, VectorType &, const Number)>
        &)> &   time_integrator,
    const std::function<void(const VectorType &, VectorType &, const Number)>
      &                                      runnable,
    const std::function<void(const Number)> &diagnostics)
  {
    unsigned int time_step_number = 1; // time step number

    // perform diagnostics of the initial condition
    diagnostics(start_time);

    // perform sequence of time steps (We multiply final_time by 1+1E-13 to
    // take care of roundoff errors)
    for (Number time = start_time + time_step;
         time <= final_time * (1.0000000000001) &&
         time_step_number <= max_time_step_number;
         time += time_step, ++time_step_number)
      {
        // perform time step
        time_integrator(solution, time - time_step, time_step, runnable);

        // perform post-processing (diagnostic, vtk output, ...)
        diagnostics(time);
      }

    return time_step_number - 1;
  }

  template class TimeLoop<double,
                          dealii::LinearAlgebra::SharedMPI::Vector<double>>;

} // namespace hyperdeal