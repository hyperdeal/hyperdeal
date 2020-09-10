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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/revision.h>

#include <hyper.deal/base/revision.h>
#include <hyper.deal/base/utilities.h>
#include <immintrin.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <vector>

namespace hyperdeal
{
  namespace Utilities
  {
    std::pair<unsigned int, unsigned int>
    lex_to_pair(const unsigned int rank,
                const unsigned int size1,
                const unsigned int size2)
    {
      AssertThrow(rank < size1 * size2, dealii::ExcMessage("Invalid rank."));
      return {rank % size1, rank / size1};
    }



    std::pair<unsigned int, unsigned int>
    decompose(const unsigned int &number)
    {
      std::vector<std::pair<unsigned int, unsigned int>> possible_solutions;

      // TODO: not optimal since N^2
      for (unsigned int i = 1; i <= number; i++)
        for (unsigned int j = 1; j <= i; j++)
          if (i * j == number)
            possible_solutions.emplace_back(i, j);

      AssertThrow(possible_solutions.size() > 0,
                  dealii::ExcMessage("No possible decomposition found!"));

      std::sort(possible_solutions.begin(),
                possible_solutions.end(),
                [](const auto &a, const auto &b) {
                  return std::abs((int)a.first - (int)a.second) <
                         std::abs((int)b.first - (int)b.second);
                });

      return possible_solutions.front();
    }



    template <typename StreamType>
    void
    print_version(const StreamType &stream)
    {
      stream << "-- hyper.deal-version " << std::endl;
      stream << "-- hyper.deal-branch: " << HYPER_DEAL_GIT_BRANCH << std::endl;
      stream << "-- hyper.deal-hash:   " << HYPER_DEAL_GIT_REVISION
             << std::endl;
      stream << "--                    " << std::endl;
      stream << "-- deal.II-version    " << std::endl;
      stream << "-- deal.II-branch:    " << DEAL_II_GIT_BRANCH << std::endl;
      stream << "-- deal.II-hash:      " << DEAL_II_GIT_REVISION << std::endl;
      stream << "--                    " << std::endl;
    }

    // template instantiations
    template void
    print_version(const dealii::ConditionalOStream &stream);

  } // namespace Utilities
} // namespace hyperdeal
