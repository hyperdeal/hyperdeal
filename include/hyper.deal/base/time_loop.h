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

#ifndef HYPERDEAL_FUNCTIONALITIES_TIME_LOOP
#define HYPERDEAL_FUNCTIONALITIES_TIME_LOOP

#include <hyper.deal/base/config.h>

#include <hyper.deal/base/time_loop_parameters.h>

#include <functional>

namespace hyperdeal
{
  /**
   * Time loop class.
   */
  template <typename Number, typename VectorType>
  class TimeLoop
  {
  public:
    /**
     * Configure time loop.
     */
    void
    reinit(const TimeLoopParamters<Number> &parameters);

    /**
     * Run time loop. For each time step, a @p time_integrator, which evaluates
     * the right-hand side function @runnable, and a postprocessor
     * @p diagnostics is called.
     */
    int
    loop(
      VectorType &solution,
      const std::function<
        void(VectorType &,
             const Number,
             const Number,
             const std::function<
               void(const VectorType &, VectorType &, const Number)> &)>
        &time_integrator,
      const std::function<void(const VectorType &, VectorType &, const Number)>
        &                                      runnable,
      const std::function<void(const Number)> &diagnostics);

    /**
     * Constant time step size.
     */
    Number time_step;

    /**
     * Start time. The default is zero, however, in the case of restarts
     * one might specify a different value.
     */
    Number start_time;

    /**
     * Final time.
     */
    Number final_time;

    /**
     * Maximal number of time steps, after which the simulation terminates
     * even if the final time has not been reached.
     */
    unsigned int max_time_step_number;
  };
} // namespace hyperdeal

#endif