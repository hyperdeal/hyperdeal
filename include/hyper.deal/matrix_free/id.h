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

#ifndef HYPERDEAL_NDIM_MATRIXFREE_ID
#define HYPERDEAL_NDIM_MATRIXFREE_ID

#include <hyper.deal/base/config.h>

namespace hyperdeal
{
  /**
   * ID describing cells and faces uniquely in phase space.
   */
  struct TensorID
  {
    /**
     * Type of face in phase space.
     */
    enum class SpaceType
    {
      XV,
      X,
      V
    };

    /**
     * Constructor.
     */
    TensorID(const unsigned int x,
             const unsigned int v,
             const unsigned int macro,
             const SpaceType    type = SpaceType::XV)
      : x(x)
      , v(v)
      , macro(macro)
      , type(type)
    {}

    /**
     * Macro-cell/face ID in x-space.
     */
    const unsigned int x;

    /**
     * Macro-cell/face ID in v-space.
     */
    const unsigned int v;

    /**
     * Macro-cell/face ID in phase-space.
     */
    const unsigned int macro;

    /**
     * Type of face in phase space.
     */
    const SpaceType type;
  };

} // namespace hyperdeal

#endif