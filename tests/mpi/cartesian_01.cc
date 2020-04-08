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


// Test shared memory functionality.

#include <deal.II/base/mpi.h>

#include <hyper.deal/base/mpi.h>
#include <hyper.deal/base/utilities.h>

#include "../tests.h"

using namespace dealii;

void
test(const MPI_Comm &comm, const unsigned int size_x, const unsigned int size_v)
{
  MPI_Comm cart_comm =
    hyperdeal::mpi::create_rectangular_comm(comm, size_x, size_v);

  if (cart_comm == MPI_COMM_NULL)
    {
      deallog << "Not part of Cartesian communicator!" << std::endl;
      return;
    }

  MPI_Comm row_comm =
    hyperdeal::mpi::create_row_comm(cart_comm, size_x, size_v);
  MPI_Comm col_comm =
    hyperdeal::mpi::create_column_comm(cart_comm, size_x, size_v);

  for (const auto &i : hyperdeal::mpi::procs_of_sm(comm, cart_comm))
    deallog << i << " ";
  deallog << std::endl;

  for (const auto &i : hyperdeal::mpi::procs_of_sm(comm, row_comm))
    deallog << i << " ";
  deallog << std::endl;

  for (const auto &i : hyperdeal::mpi::procs_of_sm(comm, col_comm))
    deallog << i << " ";
  deallog << std::endl;

  MPI_Comm_free(&col_comm);
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&cart_comm);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  test(comm, 5, 4);
}
