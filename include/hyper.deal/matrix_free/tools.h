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

#ifndef hyperdeal_matrix_free_tools_h
#define hyperdeal_matrix_free_tools_h

#include <deal.II/base/config.h>

#include <deal.II/matrix_free/matrix_free.h>

namespace hyperdeal
{
  /**
   * A namespace for utility functions in the context of matrix-free operator
   * evaluation.
   */
  namespace MatrixFreeTools
  {
    template <int dim, typename VectorizedArrayType>
    VectorizedArrayType
    evaluate_scalar_function(
      const dealii::Point<dim, VectorizedArrayType> &point,
      const dealii::Function<dim, typename VectorizedArrayType::value_type>
        &                function,
      const unsigned int n_lanes)
    {
      VectorizedArrayType result = 0;

      for (unsigned int v = 0; v < n_lanes; ++v)
        {
          dealii::Point<dim> p;
          for (unsigned int d = 0; d < dim; ++d)
            p[d] = point[d][v];
          result[v] = function.value(p);
        }

      return result;
    }


  } // namespace MatrixFreeTools


} // namespace hyperdeal


#endif
