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


// Test number of stages of time integrators.

#include <deal.II/lac/la_parallel_vector.h>

#include <hyper.deal/base/time_integrators.h>

#include "../tests.h"

int
main()
{
  initlog();

  using Number     = double;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;
  using Integrator =
    hyperdeal::LowStorageRungeKuttaIntegrator<Number, VectorType>;

  VectorType vct_Ki, vct_Ti;

  deallog << Integrator(vct_Ki, vct_Ti, "rk33").n_stages() << std::endl;
  deallog << Integrator(vct_Ki, vct_Ti, "rk45").n_stages() << std::endl;
  deallog << Integrator(vct_Ki, vct_Ti, "rk47").n_stages() << std::endl;
  deallog << Integrator(vct_Ki, vct_Ti, "rk59").n_stages() << std::endl;
}
