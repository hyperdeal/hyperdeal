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

#ifndef HYPERDEAL_MPI
#define HYPERDEAL_MPI

#include <hyper.deal/base/config.h>

#include <deal.II/base/mpi.h>

namespace hyperdeal
{
  namespace mpi
  {
    /**
     * Split up given MPI communicator @p comm in new MPI communicators
     *   only containing processes on the same shared-memory domain.
     */
    MPI_Comm
    create_sm(const MPI_Comm &comm);

    /**
     * Get size of shared-memory domain.
     *
     * @note As approximation this function takes the maximum of the sizes
     *   of all shared-memory communicators @p comm_sm in @comm.
     */
    unsigned int
    n_procs_of_sm(const MPI_Comm &comm, const MPI_Comm &comm_sm);

    /**
     * Return a list of IDs of processes living in @comm_shared.
     *
     * @note As ID the rank in @comm is used.
     */
    std::vector<unsigned int>
    procs_of_sm(const MPI_Comm &comm, const MPI_Comm &comm_shared);

    /**
     * Let rank 0 print information of all shared memory domains to the
     * screen.
     *
     * TODO: specify stream.
     *
     * TODO: add test
     */
    void
    print_sm(const MPI_Comm &comm, const MPI_Comm &comm_sm);

    /**
     * Let rank 0 print the rank in @p comm_old and @p comm_new of each
     * process to the screen.
     *
     * TODO: specify stream.
     *
     * TODO: add test
     */
    void
    print_new_order(const MPI_Comm &comm_old, const MPI_Comm &comm_new);

    /**
     * Create rectangular Cartesian communicator of @p size_x and @p size_v
     * from a given communicator @p comm.
     *
     * @note: Processes with rank >= @p size_x * @p size_v are assigned
     *  MPI_COMM_NULL, so that the user can check for
     *  if(new_comm!=MPI_COMM_NULL) and only proceed computation with processes
     *  fulfilling the condition.
     */
    MPI_Comm
    create_rectangular_comm(const MPI_Comm &   comm,
                            const unsigned int size_x,
                            const unsigned int size_v);

    /**
     * Sort processes in blocks of size @p group_size and sort the blocks
     * along a z-curve/Morton-order.
     *
     * @onte Processes within a block are enumerated lexicographically.
     *
     * @note We also call the result of this function `virtual topology`.
     */
    MPI_Comm
    create_z_order_comm(const MPI_Comm &                            comm,
                        const std::pair<unsigned int, unsigned int> procs,
                        const std::pair<unsigned int, unsigned int> group_size);

    /**
     * Create a new row communicator from a Cartesian communicator @p comm of
     * size @p size1 and @p size2.
     *
     * @note Size we use a lexicographical numbering, all processes with
     *   `rank / size1` are grouped together.
     *
     * @pre The size of @p @comm has to match the product of @p size1 and
     *   @size2.
     */
    MPI_Comm
    create_row_comm(const MPI_Comm &   comm,
                    const unsigned int size1,
                    const unsigned int size2);

    /**
     * The same as above, just for column..
     *
     * @note Size we use a lexicographical numbering, all processes with
     *   `rank % size1` are grouped together.
     */
    MPI_Comm
    create_column_comm(const MPI_Comm &   comm,
                       const unsigned int size1,
                       const unsigned int size2);

  } // namespace mpi
} // namespace hyperdeal

#endif