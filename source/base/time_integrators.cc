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

#include <hyper.deal/base/time_integrators.h>
#include <hyper.deal/lac/sm_vector.h>

namespace hyperdeal
{
  template <typename Number, typename VectorType>
  LowStorageRungeKuttaIntegrator<Number, VectorType>::
    LowStorageRungeKuttaIntegrator(VectorType &      vec_Ki,
                                   VectorType &      vec_Ti,
                                   const std::string type,
                                   const bool        only_Ti_is_ghosted)
    : vec_Ki(vec_Ki)
    , vec_Ti(vec_Ti)
    , only_Ti_is_ghosted(only_Ti_is_ghosted)
  {
    // Runge-Kutta coefficients
    // see: Kennedy, Carpenter, Lewis, 2000
    if (type == "rk33")
      {
        bi = {{0.245170287303492, 0.184896052186740, 0.569933660509768}};
        ai = {{0.755726351946097, 0.386954477304099}};
      }
    else if (type == "rk45")
      {
        bi = {{1153189308089. / 22510343858157.,
               1772645290293. / 4653164025191.,
               -1672844663538. / 4480602732383.,
               2114624349019. / 3568978502595.,
               5198255086312. / 14908931495163.}};
        ai = {{970286171893. / 4311952581923.,
               6584761158862. / 12103376702013.,
               2251764453980. / 15575788980749.,
               26877169314380. / 34165994151039.}};
      }
    else if (type == "rk47")
      {
        bi = {{0.0941840925477795334,
               0.149683694803496998,
               0.285204742060440058,
               -0.122201846148053668,
               0.0605151571191401122,
               0.345986987898399296,
               0.186627171718797670}};
        ai = {{0.241566650129646868 + bi[0],
               0.0423866513027719953 + bi[1],
               0.215602732678803776 + bi[2],
               0.232328007537583987 + bi[3],
               0.256223412574146438 + bi[4],
               0.0978694102142697230 + bi[5]}};
      }
    else if (type == "rk59")
      {
        bi = {{2274579626619. / 23610510767302.,
               693987741272. / 12394497460941.,
               -347131529483. / 15096185902911.,
               1144057200723. / 32081666971178.,
               1562491064753. / 11797114684756.,
               13113619727965. / 44346030145118.,
               393957816125. / 7825732611452.,
               720647959663. / 6565743875477.,
               3559252274877. / 14424734981077.}};
        ai = {{1107026461565. / 5417078080134.,
               38141181049399. / 41724347789894.,
               493273079041. / 11940823631197.,
               1851571280403. / 6147804934346.,
               11782306865191. / 62590030070788.,
               9452544825720. / 13648368537481.,
               4435885630781. / 26285702406235.,
               2357909744247. / 11371140753790.}};
      }
    else
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
  }

  template <typename Number, typename VectorType>
  void
  LowStorageRungeKuttaIntegrator<Number, VectorType>::perform_time_step(
    VectorType &  solution,
    const Number &current_time,
    const Number &time_step,
    const std::function<void(const VectorType &, VectorType &, const Number)>
      &op)
  {
    // cache local size
    const unsigned int size = solution.locally_owned_size();

    // definition of a stage
    auto perform_stage = [&](const Number      current_time,
                             const Number      factor_solution,
                             const Number      factor_ai,
                             const VectorType &current_Ti,
                             VectorType &      vec_Ki,
                             VectorType &      solution,
                             VectorType &      next_Ti) {
      // call operator
      op(current_Ti, vec_Ki, current_time);

      const Number ai = factor_ai;
      const Number bi = factor_solution;
      if (ai == Number())
        {
          for (unsigned int i = 0; i < size; ++i)
            {
              const Number K_i          = vec_Ki.local_element(i);
              const Number sol_i        = solution.local_element(i);
              solution.local_element(i) = sol_i + bi * K_i;
            }
        }
      else
        {
          for (unsigned int i = 0; i < size; ++i)
            {
              const Number K_i          = vec_Ki.local_element(i);
              const Number sol_i        = solution.local_element(i);
              solution.local_element(i) = sol_i + bi * K_i;
              next_Ti.local_element(i)  = sol_i + ai * K_i;
            }
        }
    };


    // perform first stage
    if (only_Ti_is_ghosted)
      {
        // swap solution and Ti
        for (unsigned int i = 0; i < size; i++)
          vec_Ti.local_element(i) = solution.local_element(i);

        perform_stage(current_time,
                      bi[0] * time_step,
                      ai[0] * time_step,
                      vec_Ti,
                      vec_Ki,
                      solution,
                      vec_Ti);
      }
    else
      {
        perform_stage(current_time,
                      bi[0] * time_step,
                      ai[0] * time_step,
                      solution,
                      vec_Ti,
                      solution,
                      vec_Ti);
      }


    // perform rest stages
    Number sum_previous_bi = 0;
    for (unsigned int stage = 1; stage < bi.size(); ++stage)
      {
        const Number c_i = sum_previous_bi + ai[stage - 1];
        perform_stage(current_time + c_i * time_step,
                      bi[stage] * time_step,
                      (stage == bi.size() - 1 ? 0 : ai[stage] * time_step),
                      vec_Ti,
                      vec_Ki,
                      solution,
                      vec_Ti);
        sum_previous_bi += bi[stage - 1];
      }
  }

  template <typename Number, typename VectorType>
  unsigned int
  LowStorageRungeKuttaIntegrator<Number, VectorType>::n_stages() const
  {
    return bi.size();
  }


  template class LowStorageRungeKuttaIntegrator<
    double,
    dealii::LinearAlgebra::SharedMPI::Vector<double>>;

} // namespace hyperdeal
