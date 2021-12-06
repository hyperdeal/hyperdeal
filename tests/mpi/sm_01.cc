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

#include <deal.II/tests/tests.h>

#include <hyper.deal/base/mpi.h>
#include <hyper.deal/base/utilities.h>

using namespace dealii;

void
test(const MPI_Comm &comm)
{
  const unsigned int rank = dealii::Utilities::MPI::this_mpi_process(comm);

  MPI_Comm new_comm;
  MPI_Comm_split(comm, rank < 3, rank, &new_comm);

  deallog << hyperdeal::mpi::n_procs_of_sm(comm, new_comm) << std::endl;

  for (const auto &i : hyperdeal::mpi::procs_of_sm(comm, new_comm))
    deallog << i << " ";
  deallog << std::endl;

  MPI_Comm_free(&new_comm);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  test(comm);
}
