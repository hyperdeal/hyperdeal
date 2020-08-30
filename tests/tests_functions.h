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

#ifndef HYPERDEAL_TESTS_FUNCTIONS
#define HYPERDEAL_TESTS_FUNCTIONS

#include <deal.II/base/function.h>

template <int DIM, typename Number>
class SinusConsinusFunction : public dealii::Function<DIM, Number>
{
public:
  SinusConsinusFunction()
    : dealii::Function<DIM, Number>(1)
  {}

  virtual Number
  value(const dealii::Point<DIM> &p, const unsigned int = 1) const
  {
    Number result = std::sin(p[0] * dealii::numbers::PI);

    for (unsigned int d = 1; d < DIM; ++d)
      result *= std::cos(p[d] * dealii::numbers::PI);

    return result;
  }
};

#endif