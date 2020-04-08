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

#include <hyper.deal/base/memory_consumption.h>

#include "../tests.h"

using namespace dealii;

void
test(const MPI_Comm &comm)
{
  hyperdeal::MemoryConsumption m1("m1", 1000);
  hyperdeal::MemoryConsumption m2("m2");
  m2.insert(m1);
  m2.insert("m3", 9000);

  m1.print(comm, deallog);
  deallog << std::endl;

  m2.print(comm, deallog);
  deallog << std::endl;

  m1.print(comm, deallog);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  test(comm);
}
