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

#ifndef HYPERDEAL_FUNCTIONALITIES_UTIL
#define HYPERDEAL_FUNCTIONALITIES_UTIL

#include <hyper.deal/base/config.h>

#include <deal.II/base/exceptions.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace hyperdeal
{
  namespace Utilities
  {
    /**
     * Given an @id and the sizes of a Cartesian system @size1 and @size2,
     * return the position in 2D space.
     *
     * @note A lexicographical numbering is used.
     */
    std::pair<unsigned int, unsigned int>
    lex_to_pair(const unsigned int id,
                const unsigned int size1,
                const unsigned int size2);

    /**
     * Factorize @p number into number=number1*number2 with the following
     * properties:
     *  - min(number1 - number2) and
     *  - number1 >= number2
     */
    std::pair<unsigned int, unsigned int>
    decompose(const unsigned int &number);

    /**
     * Print hyper.deal and deal.II git version information.
     */
    template <typename StreamType>
    void
    print_version(const StreamType &stream);
  } // namespace Utilities
} // namespace hyperdeal

#endif
