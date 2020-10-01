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

#include <deal.II/base/mpi.templates.h>

#include <hyper.deal/base/mpi.h>
#include <hyper.deal/base/utilities.h>
#include <immintrin.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <vector>

namespace hyperdeal
{
  namespace mpi
  {
    MPI_Comm
    create_sm(const MPI_Comm &comm)
    {
      int rank;
      MPI_Comm_rank(comm, &rank);

      MPI_Comm comm_shared;
      MPI_Comm_split_type(
        comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &comm_shared);

      return comm_shared;
    }



    unsigned int
    n_procs_of_sm(const MPI_Comm &comm, const MPI_Comm &comm_sm)
    {
      // determine size of current shared memory communicator
      int size_shared;
      MPI_Comm_size(comm_sm, &size_shared);

      // determine maximum, since some shared memory communicators might not be
      // filed completely
      int size_shared_max;
      MPI_Allreduce(&size_shared, &size_shared_max, 1, MPI_INT, MPI_MAX, comm);

      return size_shared_max;
    }



    std::vector<unsigned int>
    procs_of_sm(const MPI_Comm &comm, const MPI_Comm &comm_shared)
    {
      // extract information from comm
      int rank_;
      MPI_Comm_rank(comm, &rank_);

      const unsigned int rank = rank_;

      // extract information from sm-comm
      int size_shared;
      MPI_Comm_size(comm_shared, &size_shared);

      // gather ranks
      std::vector<unsigned int> ranks_shared(size_shared);
      MPI_Allgather(
        &rank, 1, MPI_UNSIGNED, ranks_shared.data(), 1, MPI_INT, comm_shared);

      return ranks_shared;
    }



    template <typename T>
    std::vector<std::vector<T>>
    allgatherv(const std::vector<T> data_in, const MPI_Comm &comm)
    {
      int size;
      MPI_Comm_size(comm, &size);

      std::vector<int> recvcounts(size);

      int data_size = data_in.size();
      MPI_Allgather(
        &data_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

      std::vector<int> displs(size + 1);

      displs[0] = 0;
      for (int i = 0; i < size; i++)
        displs[i + 1] = displs[i] + recvcounts[i];


      std::vector<T> temp(displs.back());
      MPI_Allgatherv(data_in.data(),
                     data_in.size(),
                     dealii::Utilities::MPI::internal::mpi_type_id(&temp[0]),
                     temp.data(),
                     recvcounts.data(),
                     displs.data(),
                     dealii::Utilities::MPI::internal::mpi_type_id(&temp[0]),
                     comm);


      std::vector<std::vector<T>> result(size);
      for (int i = 0; i < size; i++)
        result[i] = std::vector<T>(temp.begin() + displs[i],
                                   temp.begin() + displs[i + 1]);

      return result;
    }



    void
    print_sm(const MPI_Comm &comm, const MPI_Comm &comm_sm)
    {
      int rank, rank_sm;
      MPI_Comm_rank(comm, &rank);

      MPI_Comm_rank(comm_sm, &rank_sm);

      const auto procs_of_sm_ = procs_of_sm(comm, comm_sm);

      MPI_Comm comm_sm_index;
      MPI_Comm_split(comm, rank_sm == 0, rank, &comm_sm_index);

      if (rank_sm == 0)
        {
          const auto list = allgatherv(procs_of_sm_, comm_sm_index);

          if (rank == 0)
            for (unsigned int i = 0; i < list.size(); i++)
              {
                for (unsigned int j = 0; j < list[i].size(); j++)
                  printf("%5d ", list[i][j]);
                printf("\n");
              }
        }

      MPI_Comm_free(&comm_sm_index);
    }



    void
    print_new_order(const MPI_Comm &comm_old, const MPI_Comm &comm_new)
    {
      int size, rank, new_number;
      MPI_Comm_rank(comm_old, &rank);
      MPI_Comm_size(comm_old, &size);

      if (comm_new == MPI_COMM_NULL)
        new_number = -1;
      else
        MPI_Comm_rank(comm_new, &new_number);

      std::vector<int> recv_data(size);
      MPI_Gather(
        &new_number, 1, MPI_INT, recv_data.data(), 1, MPI_INT, 0, comm_old);

      if (rank == 0)
        for (unsigned int i = 0; i < recv_data.size(); i++)
          printf("(%5d,%5d)\n", i, recv_data[i]);
    }



    MPI_Comm
    create_row_comm(const MPI_Comm &   comm,
                    const unsigned int size1,
                    const unsigned int size2)
    {
      int size, rank;
      MPI_Comm_size(comm, &size);
      AssertThrow(static_cast<unsigned int>(size) == size1 * size2,
                  dealii::ExcMessage("Invalid communicator size."));

      MPI_Comm_rank(comm, &rank);

      MPI_Comm row_comm;
      MPI_Comm_split(comm,
                     Utilities::lex_to_pair(rank, size1, size2).second,
                     rank,
                     &row_comm);
      return row_comm;
    }



    MPI_Comm
    create_column_comm(const MPI_Comm &   comm,
                       const unsigned int size1,
                       const unsigned int size2)
    {
      int size, rank;
      MPI_Comm_size(comm, &size);
      AssertThrow(static_cast<unsigned int>(size) == size1 * size2,
                  dealii::ExcMessage("Invalid communicator size."));

      MPI_Comm_rank(comm, &rank);

      MPI_Comm col_comm;
      MPI_Comm_split(comm,
                     Utilities::lex_to_pair(rank, size1, size2).first,
                     rank,
                     &col_comm);
      return col_comm;
    }



    MPI_Comm
    create_rectangular_comm(const MPI_Comm &   comm,
                            const unsigned int size_x,
                            const unsigned int size_v)
    {
      int rank, size;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &size);

      AssertThrow((size_x * size_v) <= static_cast<unsigned int>(size),
                  dealii::ExcMessage("Not enough ranks."));

      MPI_Comm sub_comm;
      MPI_Comm_split(comm,
                     (static_cast<unsigned int>(rank) < (size_x * size_v)),
                     rank,
                     &sub_comm);

      if (static_cast<unsigned int>(rank) < (size_x * size_v))
        return sub_comm;
      else
        {
          MPI_Comm_free(&sub_comm);
          return MPI_COMM_NULL;
        }
    }



#ifndef __AVX2__
    unsigned int
    _pdep_u32(unsigned int a, unsigned int mask)
    {
      // Intrinsics command only available from AVX2 upwards. This is a
      // workaround for older processor architectures - according to:
      // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_pdep_u32&expand=4151

      unsigned int dst = 0;
      unsigned int tmp = a;
      unsigned int k   = 0;

      for (unsigned int m = 0; m < 32; m++)
        if ((mask >> m & 1) == 1)
          dst += ((tmp >> k++) & 1) << m;

      return dst;
    }
#endif



    MPI_Comm
    create_z_order_comm(const MPI_Comm &                            comm,
                        const std::pair<unsigned int, unsigned int> procs,
                        const std::pair<unsigned int, unsigned int> group_size)
    {
      const unsigned int size_x = procs.first;
      const unsigned int size_v = procs.second;

      const unsigned int group_size_x = group_size.first;
      const unsigned int group_size_v = group_size.second;

      int size, rank;
      MPI_Comm_size(comm, &size);
      MPI_Comm_rank(comm, &rank);
      AssertThrow(static_cast<unsigned int>(size) == size_x * size_v,
                  dealii::ExcMessage("Invalid communicator size."));

      auto new_number = [&](const int i) {
        auto xy_to_morton = [](uint32_t x, uint32_t y) {
          return _pdep_u32(x, 0x55555555) | _pdep_u32(y, 0xaaaaaaaa);
        };

        const unsigned int x = i % size_x;
        const unsigned int v = i / size_x;
        const unsigned int new_number =
          xy_to_morton(x / group_size_x, v / group_size_v) * group_size_x *
            group_size_v +
          (v % group_size_v) * group_size_x + (x % group_size_x);

        return new_number + ((x >= (size_x / group_size_x) * group_size_x ||
                              v >= (size_v / group_size_v) * group_size_v) ?
                               size_x * size_v :
                               0);
      };

      MPI_Comm new_comm;

      std::vector<std::pair<unsigned int, unsigned int>> pairs;

      for (unsigned int v = 0; v < size_v; v++)
        for (unsigned int x = 0; x < size_x; x++)
          pairs.emplace_back(x + v * size_x, new_number(x + v * size_x));

      std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
        return a.second < b.second;
      });

      MPI_Comm_split(comm, 0, pairs[rank].first, &new_comm);

      return new_comm;
    }

  } // namespace mpi
} // namespace hyperdeal