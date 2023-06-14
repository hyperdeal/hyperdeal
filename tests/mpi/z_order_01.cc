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


// Update ghost values with the help of LinearAlgebra::distributed::Partitioner.

#include <deal.II/base/mpi.h>

#include <hyper.deal/base/mpi.h>
#include <hyper.deal/base/utilities.h>

#include "../tests.h"

using namespace dealii;

void
test(const MPI_Comm &comm, const std::pair<int, int> &group_size)
{
  const auto pair = hyperdeal::Utilities::decompose(
    dealii::Utilities::MPI::n_mpi_processes(comm));

  MPI_Comm comm_z = hyperdeal::mpi::create_z_order_comm(comm, pair, group_size);

  auto procs = hyperdeal::mpi::procs_of_sm(comm, comm_z);

  if (dealii::Utilities::MPI::this_mpi_process(comm) != 0)
    return;

  for (unsigned int i = 0, c = 0; i < pair.second; i++)
    {
      for (unsigned int j = 0; j < pair.first; j++)
        deallog << std::setw(5) << procs[c++];
      deallog << std::endl;
    }
  deallog << std::endl;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  // 28 = 7 * 4
  deallog.push("version1");
  test(comm, {1, 1});
  deallog.pop();
  deallog.push("version2");
  test(comm, {2, 2});
  deallog.pop();
  deallog.push("version3");
  test(comm, {3, 2});
  deallog.pop();
  deallog.push("version4");
  test(comm, {2, 3});
  deallog.pop();
}
