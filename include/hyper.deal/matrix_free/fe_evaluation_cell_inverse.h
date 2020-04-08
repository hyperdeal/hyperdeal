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

#ifndef HYPERDEAL_NDIM_FEEVALUATION_CELL_INVERSE
#define HYPERDEAL_NDIM_FEEVALUATION_CELL_INVERSE

#include <hyper.deal/base/config.h>

#include <hyper.deal/matrix_free/fe_evaluation_cell.h>

namespace hyperdeal
{
  /**
   * The same as FEEvaluation but specialized for inverse mass matrix.
   *
   * TODO: rename?
   */
  template <int dim_x,
            int dim_v,
            int degree,
            typename Number,
            typename VectorizedArrayType>
  class FEEvaluationInverse : public FEEvaluation<dim_x,
                                                  dim_v,
                                                  degree,
                                                  degree + 1,
                                                  Number,
                                                  VectorizedArrayType>
  {
  public:
    using PARENT = FEEvaluation<dim_x,
                                dim_v,
                                degree,
                                degree + 1,
                                Number,
                                VectorizedArrayType>;

    /**
     * Constructor.
     */
    FEEvaluationInverse(
      const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &matrix_free,
      const unsigned int                                           dof_no_x,
      const unsigned int                                           dof_no_v,
      const unsigned int                                           quad_no_x,
      const unsigned int                                           quad_no_v);

    /**
     * Return inverse shape function.
     */
    const dealii::AlignedVector<VectorizedArrayType> *
    get_inverse_shape() const;

    /**
     * Return product of @p data and 1/(|J|xw).
     */
    inline DEAL_II_ALWAYS_INLINE //
      void
      submit_inv(VectorizedArrayType *__restrict data_ptr,
                 const unsigned int q,
                 const unsigned int q1,
                 const unsigned int q2);

  private:
    /**
     * Reference to the inverse shape function.
     */
    const dealii::AlignedVector<VectorizedArrayType> &inverse_shape;
  };



  template <int dim_x, int dim_v, int degree, typename Number, typename VNumber>
  FEEvaluationInverse<dim_x, dim_v, degree, Number, VNumber>::
    FEEvaluationInverse(
      const MatrixFree<dim_x, dim_v, Number, VNumber> &matrix_free,
      const unsigned int                               dof_no_x,
      const unsigned int                               dof_no_v,
      const unsigned int                               quad_no_x,
      const unsigned int                               quad_no_v)
    : PARENT(matrix_free, dof_no_x, dof_no_v, quad_no_x, quad_no_v)
    , inverse_shape(
        this->phi_x.get_shape_info().data[0].inverse_shape_values_eo)
  {}



  template <int dim_x, int dim_v, int degree, typename Number, typename VNumber>
  const dealii::AlignedVector<VNumber> *
  FEEvaluationInverse<dim_x, dim_v, degree, Number, VNumber>::
    get_inverse_shape() const
  {
    return &inverse_shape;
  }



  template <int dim_x, int dim_v, int degree, typename Number, typename VNumber>
  inline DEAL_II_ALWAYS_INLINE //
    void
    FEEvaluationInverse<dim_x, dim_v, degree, Number, VNumber>::submit_inv(
      VNumber *__restrict data_ptr,
      const unsigned int q,
      const unsigned int q1,
      const unsigned int q2)
  {
    data_ptr[q] /=
      this->phi_x.JxW(q1) *
      this->phi_v.JxW(q2)[PARENT::n_vectors_v == 1 ? 0 : this->lane_y];
  }

} // namespace hyperdeal

#endif
