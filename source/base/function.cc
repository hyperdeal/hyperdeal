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

// instantiate functions for higher dimensions since in
// dealii/source/base/function.inst.in it is done only for 1-3

#include <deal.II/base/function.templates.h>

DEAL_II_NAMESPACE_OPEN

template class Function<4, double>;
template class Function<5, double>;
template class Function<6, double>;

namespace Functions
{
  template class ZeroFunction<4, double>;
  template class ZeroFunction<5, double>;
  template class ZeroFunction<6, double>;
} // namespace Functions

DEAL_II_NAMESPACE_CLOSE
