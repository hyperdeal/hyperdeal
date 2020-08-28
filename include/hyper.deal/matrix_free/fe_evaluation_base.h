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

#ifndef HYPERDEAL_NDIM_FEEVALUATION_BASE
#define HYPERDEAL_NDIM_FEEVALUATION_BASE

#include <hyper.deal/base/config.h>

#include <hyper.deal/matrix_free/matrix_free.h>

namespace hyperdeal
{
  /**
   * Base class of FEEvaluation and FEFaceEvaluation.
   */
  template <int dim_x,
            int dim_v,
            int degree,
            int n_points,
            typename Number,
            typename VectorizedArrayType>
  class FEEvaluationBase
  {
  public:
    static const int dim = dim_x + dim_v;

    using NUMBER_     = Number;
    using VEC_NUMBER_ = VectorizedArrayType;
    using MF          = MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>;

    static const unsigned int n_vectors   = MF::VectorizedArrayTypeX::size();
    static const unsigned int n_vectors_v = MF::VectorizedArrayTypeV::size();

    static const int static_dofs =
      dealii::Utilities::pow((degree + 1 > n_points) ? (degree + 1) : n_points,
                             dim);

    using VectorizedArrayTypeX = typename MF::VectorizedArrayTypeX;
    using VectorizedArrayTypeV = typename MF::VectorizedArrayTypeV;

    /**
     * Constructor.
     */
    FEEvaluationBase(
      const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &matrix_free);

    /**
     * Set the view to the current cell.
     */
    virtual void
    reinit(typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::ID
             cell_index);

    /**
     * Return pointer to the internal buffer.
     */
    VectorizedArrayType *
    get_data_ptr();

    /**
     * Return shape values.
     */
    const dealii::AlignedVector<VectorizedArrayType> *
    get_shape_values() const;

    /**
     * Return gradient of the shape values.
     */
    const dealii::AlignedVector<VectorizedArrayType> *
    get_shape_gradients() const;

  protected:
    /**
     * Local storage for values and derivatives.
     */
    dealii::AlignedVector<VectorizedArrayType> data;

    /**
     * Reference to the phase-space matrix-free instance.
     */
    const MF &matrix_free;

    /**
     * Reference to the x-space matrix-free instance.
     */
    const dealii::MatrixFree<dim_x, Number, VectorizedArrayTypeX>
      &matrix_free_x;

    /**
     * Reference to the v-space matrix-free instance.
     */
    const dealii::MatrixFree<dim_v, Number, VectorizedArrayTypeV>
      &matrix_free_v;

    // information about the current cell/face (interpretation depends on
    // FEEvaluation/FEFaceEvaluation)
    unsigned int macro_cell_x;
    unsigned int macro_cell_v;
    unsigned int lane_y;
    unsigned int macro;

    /**
     * Pointer to the shape functions.
     */
    const dealii::AlignedVector<VectorizedArrayType> *shape_values;

    /**
     * Pointer to the gradient of the shape functions.
     */
    const dealii::AlignedVector<VectorizedArrayType> *shape_gradients;
  };



  template <int dim_x,
            int dim_v,
            int degree,
            int n_points,
            typename Number,
            typename VectorizedArrayType>
  FEEvaluationBase<dim_x,
                   dim_v,
                   degree,
                   n_points,
                   Number,
                   VectorizedArrayType>::
    FEEvaluationBase(
      const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &matrix_free)
    : matrix_free(matrix_free)
    , matrix_free_x(matrix_free.get_matrix_free_x())
    , matrix_free_v(matrix_free.get_matrix_free_v())
  {
    this->data.resize(static_dofs);
  }



  template <int dim_x,
            int dim_v,
            int degree,
            int n_points,
            typename Number,
            typename VectorizedArrayType>
  void
  FEEvaluationBase<dim_x,
                   dim_v,
                   degree,
                   n_points,
                   Number,
                   VectorizedArrayType>::
    reinit(typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::ID
             cell_index)
  {
    this->macro_cell_x = cell_index.x;
    this->macro_cell_v = cell_index.v / n_vectors_v;
    this->lane_y       = cell_index.v % n_vectors_v;
    this->macro        = cell_index.macro;

    Assert(this->lane_y < this->n_vectors_v,
           dealii::ExcIndexRange(this->lane_y, 0, n_vectors_v));
  }



  template <int dim_x,
            int dim_v,
            int degree,
            int n_points,
            typename Number,
            typename VectorizedArrayType>
  VectorizedArrayType *
  FEEvaluationBase<dim_x,
                   dim_v,
                   degree,
                   n_points,
                   Number,
                   VectorizedArrayType>::get_data_ptr()
  {
    return &data[0];
  }



  template <int dim_x,
            int dim_v,
            int degree,
            int n_points,
            typename Number,
            typename VectorizedArrayType>
  const dealii::AlignedVector<VectorizedArrayType> *
  FEEvaluationBase<dim_x,
                   dim_v,
                   degree,
                   n_points,
                   Number,
                   VectorizedArrayType>::get_shape_values() const
  {
    return shape_values;
  }



  template <int dim_x,
            int dim_v,
            int degree,
            int n_points,
            typename Number,
            typename VectorizedArrayType>
  const dealii::AlignedVector<VectorizedArrayType> *
  FEEvaluationBase<dim_x,
                   dim_v,
                   degree,
                   n_points,
                   Number,
                   VectorizedArrayType>::get_shape_gradients() const
  {
    return shape_gradients;
  }

} // namespace hyperdeal

#endif
