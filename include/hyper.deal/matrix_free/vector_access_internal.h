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

#ifndef HYPERDEAL_MATRIX_FREE_VECTOR_ACCESS_INTERNAL
#define HYPERDEAL_MATRIX_FREE_VECTOR_ACCESS_INTERNAL

#include <hyper.deal/base/config.h>

namespace hyperdeal
{
  namespace internal
  {
    namespace MatrixFreeFunctions
    {
      template <typename Number, typename VectorizedArrayType>
      struct VectorReader
      {
        void
        process_dof(const Number &global, Number &local) const
        {
          local = global;
        }

        void
        process_dofs_vectorized_transpose(
          const unsigned int dofs_per_cell,
          const std::array<Number *, VectorizedArrayType::size()> &global_ptr,
          VectorizedArrayType *                                    local) const
        {
          vectorized_load_and_transpose(dofs_per_cell, global_ptr, local);
        }
      };



      template <typename Number, typename VectorizedArrayType>
      struct VectorDistributorLocalToGlobal
      {
        void
        process_dof(Number &global, const Number &local) const
        {
          global += local;
        }

        void
        process_dofs_vectorized_transpose(
          const unsigned int                                 dofs_per_cell,
          std::array<Number *, VectorizedArrayType::size()> &global_ptr,
          const VectorizedArrayType *                        local) const
        {
          vectorized_transpose_and_store(true,
                                         dofs_per_cell,
                                         local,
                                         global_ptr);
        }
      };



      template <typename Number, typename VectorizedArrayType>
      struct VectorSetter
      {
        void
        process_dof(Number &global, const Number &local) const
        {
          global = local;
        }

        void
        process_dofs_vectorized_transpose(
          const unsigned int                                 dofs_per_cell,
          std::array<Number *, VectorizedArrayType::size()> &global_ptr,
          const VectorizedArrayType *                        local) const
        {
          vectorized_transpose_and_store(false,
                                         dofs_per_cell,
                                         local,
                                         global_ptr);
        }
      };

    } // namespace MatrixFreeFunctions
  }   // namespace internal
} // namespace hyperdeal

#endif
