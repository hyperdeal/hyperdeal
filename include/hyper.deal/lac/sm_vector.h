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

#ifndef HYPERDEAL_LINEARALGEBRA_SHAREDMPI_VECTOR
#define HYPERDEAL_LINEARALGEBRA_SHAREDMPI_VECTOR

#include <deal.II/lac/la_parallel_vector.h>

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace SharedMPI
  {
    template <typename Number, typename MemorySpace = MemorySpace::Host>
    using Vector =
      dealii::LinearAlgebra::distributed::Vector<Number, MemorySpace>;
  } // namespace SharedMPI
} // namespace LinearAlgebra


DEAL_II_NAMESPACE_CLOSE

#endif
