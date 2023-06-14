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

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <hyper.deal/base/time_integrators.h>
#include <hyper.deal/base/time_integrators.templates.h>

namespace hyperdeal
{
  template class LowStorageRungeKuttaIntegrator<
    double,
    dealii::LinearAlgebra::distributed::Vector<double>>;
  template class LowStorageRungeKuttaIntegrator<
    double,
    dealii::LinearAlgebra::distributed::BlockVector<double>>;

} // namespace hyperdeal
