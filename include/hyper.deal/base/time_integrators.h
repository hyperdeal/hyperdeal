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

#ifndef HYPERDEAL_FUNCTIONALITIES_TIME_INGEGRATORS
#define HYPERDEAL_FUNCTIONALITIES_TIME_INGEGRATORS

#include <hyper.deal/base/config.h>

#include <deal.II/base/exceptions.h>

#include <functional>
#include <string>
#include <vector>

namespace hyperdeal
{
  /**
   * Efficient specialized low-storage Runge-Kutta implementations.
   *
   * We provide an implementation, which only needs one vector (vec_Ti) to be
   * ghosted. This is in particular useful in high-dimensions, where
   * memory is scarce.
   *
   * @note For details on the basic low-storage implementation, see step-69
   *   in deal.II.
   */
  template <typename Number, typename VectorType>
  class LowStorageRungeKuttaIntegrator
  {
  public:
    /**
     * Constructor. The user provides from outside two register vectors
     * @p vec_Ki and @p vev_Ti.
     *
     * @note Currently rk33, rk45, rk47, rk59 (order, stages) are supported.
     */
    LowStorageRungeKuttaIntegrator(VectorType &      vec_Ki,
                                   VectorType &      vec_Ti,
                                   const std::string type,
                                   const bool        only_Ti_is_ghosted = true);

    /**
     * Perform time step: evaluate right-hand side provided by @p op at
     * a specified time @p current_time and with a given @p time_step. The
     * previous solution is provided by @p solution and the new solution
     * is written into the same vector.
     */
    void
    perform_time_step(
      VectorType &  solution,
      const Number &current_time,
      const Number &time_step,
      const std::function<void(const VectorType &, VectorType &, const Number)>
        &op);

    unsigned int
    n_stages() const;

  private:
    /**
     * First register (does not have to be ghosted - see the comments in the
     * constructor).
     */
    VectorType &vec_Ki;

    /**
     * Second register (has to be ghosted).
     */
    VectorType &vec_Ti;

    /**
     * Is only vector Ti ghosted or are all vectors ghosted.
     */
    const bool only_Ti_is_ghosted;

    /**
     * Coefficients of the Runge-Kutta stages.
     */
    std::vector<Number> ai, bi;
  };


} // namespace hyperdeal

#endif
