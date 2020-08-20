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

#ifndef HYPERDEAL_BASE_MPI_TAGS
#define HYPERDEAL_BASE_MPI_TAGS

#include <hyper.deal/base/config.h>

#include <deal.II/base/mpi.h>

namespace hyperdeal
{
  namespace mpi
  {
    namespace internal
    {
      namespace Tags
      {
        enum enumeration : std::uint16_t
        {
          // MatrixFree::loop_cell_centric() -> export
          matrix_free_loop_cell_centric_export,

          // MatrixFree::loop() -> export
          matrix_free_loop_export,

          // MatrixFree::loop() -> import
          matrix_free_loop_import,

        };
      }
    } // namespace internal
  }   // namespace mpi
} // namespace hyperdeal

#endif
