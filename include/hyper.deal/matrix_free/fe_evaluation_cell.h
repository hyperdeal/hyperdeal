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

#ifndef HYPERDEAL_NDIM_FEEVALUATION_CELL
#define HYPERDEAL_NDIM_FEEVALUATION_CELL

#include <hyper.deal/base/config.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <hyper.deal/matrix_free/fe_evaluation_base.h>
#include <hyper.deal/matrix_free/read_write_operation.h>
#include <hyper.deal/matrix_free/vector_access_internal.h>

namespace hyperdeal
{
  /**
   * The class that provides all functions necessary to evaluate functions at
   * quadrature points and cell integrations in phase space. It delegates the
   * the actual evaluation task to two dealii::FEEvaluation objects and combines
   * the result on-the-fly.
   */
  template <int dim_x,
            int dim_v,
            int degree,
            int n_points,
            typename Number,
            typename VectorizedArrayType>
  class FEEvaluation : public FEEvaluationBase<dim_x,
                                               dim_v,
                                               degree,
                                               n_points,
                                               Number,
                                               VectorizedArrayType>
  {
  public:
    // clang-format off
    static const unsigned int dim                    = dim_x + dim_v;
    static const unsigned int static_dofs_per_cell_x = dealii::Utilities::pow(degree + 1, dim_x);
    static const unsigned int static_dofs_per_cell_v = dealii::Utilities::pow(degree + 1, dim_v);
    static const unsigned int static_dofs_per_cell   = static_dofs_per_cell_x * static_dofs_per_cell_v;
    static const unsigned int n_q_points_x           = dealii::Utilities::pow(n_points, dim_x);
    static const unsigned int n_q_points_v           = dealii::Utilities::pow(n_points, dim_v);
    static const unsigned int n_q_points             = n_q_points_x * n_q_points_v;
    // clang-format on

    using PARENT = FEEvaluationBase<dim_x,
                                    dim_v,
                                    degree,
                                    n_points,
                                    Number,
                                    VectorizedArrayType>;

    static const unsigned int n_vectors   = PARENT::n_vectors;
    static const unsigned int n_vectors_v = PARENT::n_vectors_v;

    /**
     * Constructor.
     *
     * @param matrix_free Data object that contains all data.
     * @param dof_no_x    If x-space matrix_free of matrix_free was set up with
     *                    multiple DoFHandler objects, this parameter selects to
     *                    which DoFHandler/AffineConstraints pair the given
     *                    evaluator should be attached to.
     * @param dof_no_v    Same as above but for v-space.
     * @param quad_no_x   If x-space matrix_free of matrix_free was set up with
     *                    multiple Quadrature objects, this parameter selects
     *                    the appropriate number of the quadrature formula.
     * @param quad_no_v   Same as above but for v-space.
     */
    FEEvaluation(
      const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &matrix_free,
      const unsigned int                                           dof_no_x,
      const unsigned int                                           dof_no_v,
      const unsigned int                                           quad_no_x,
      const unsigned int                                           quad_no_v)
      : PARENT(matrix_free)
      , phi_x(this->matrix_free_x, dof_no_x, quad_no_x)
      , phi_v(this->matrix_free_v, dof_no_v, quad_no_v)
    {
      this->shape_values = &phi_x.get_shape_info().data[0].shape_values_eo;
      this->shape_gradients =
        &phi_x.get_shape_info().data[0].shape_gradients_collocation_eo;
    }


    /**
     * Set the view to the current cell.
     */
    void
    reinit(typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::ID
             cell_index)
    {
      PARENT::reinit(cell_index);
      phi_x.reinit(this->macro_cell_x);
      phi_v.reinit(this->macro_cell_v);
    }


    /**
     * Read values from @p src vector and write it into the buffer @p data.
     */
    void
    read_dof_values(const dealii::LinearAlgebra::SharedMPI::Vector<Number> &src,
                    VectorizedArrayType *data) const
    {
      internal::MatrixFreeFunctions::ReadWriteOperation<Number>(
        this->matrix_free.get_dof_info(),
        this->matrix_free.get_face_info(),
        this->matrix_free.get_shape_info())
        .template process_cell<dim, degree>(
          internal::MatrixFreeFunctions::VectorReader<Number,
                                                      VectorizedArrayType>(),
          src.other_values(),
          data,
          this->macro);
    }



    /**
     * Read values from @p src vector and write them into the internal buffer.
     */
    void
    read_dof_values(const dealii::LinearAlgebra::SharedMPI::Vector<Number> &src)
    {
      read_dof_values(src, &this->data[0]);
    }



    /**
     * Sum values from the internal buffer into the vector @p dst.
     */
    void
    set_dof_values(dealii::LinearAlgebra::SharedMPI::Vector<Number> &dst) const
    {
      internal::MatrixFreeFunctions::ReadWriteOperation<Number>(
        this->matrix_free.get_dof_info(),
        this->matrix_free.get_face_info(),
        this->matrix_free.get_shape_info())
        .template process_cell<dim, degree>(
          internal::MatrixFreeFunctions::VectorSetter<Number,
                                                      VectorizedArrayType>(),
          dst.other_values(),
          &this->data[0],
          this->macro);
    }



    /**
     * Return number of filled lanes.
     */
    inline unsigned int
    n_vectorization_lanes_filled() const
    {
      return this->matrix_free_x.n_active_entries_per_cell_batch(
        this->macro_cell_x);
    }


    /**
     * Return product of @p data and |J|xw.
     *
     * TODO: remove.
     */
    inline VectorizedArrayType
    submit_inplace(const VectorizedArrayType data, const unsigned int q) const
    {
      return submit_inplace(data, q, q % n_q_points_x, q / n_q_points_x);
    }



    /**
     * Return product of @p data and |J|xw.
     *
     * TODO: remove.
     */
    inline VectorizedArrayType
    submit_inplace(const VectorizedArrayType data,
                   unsigned int /*q*/,
                   unsigned int qx,
                   unsigned int qv) const
    {
      return data * phi_x.JxW(qx) *
             phi_v.JxW(qv)[n_vectors_v == 1 ? 0 : this->lane_y];
    }


    inline DEAL_II_ALWAYS_INLINE //
      VectorizedArrayType
      JxW(unsigned int qx, unsigned int qv) const
    {
      return phi_x.JxW(qx) * phi_v.JxW(qv)[n_vectors_v == 1 ? 0 : this->lane_y];
    }



    /**
     * Submit gradient in x-space
     */
    inline DEAL_II_ALWAYS_INLINE //
      void
      submit_gradient_x(VectorizedArrayType *__restrict data_ptr_out,
                        const VectorizedArrayType *__restrict grad_in,
                        const unsigned int q,
                        const unsigned int qx,
                        const unsigned int qv) const
    {
      const auto jxw =
        phi_x.JxW(qx) * phi_v.JxW(qv)[n_vectors_v == 1 ? 0 : this->lane_y];
      const auto jacobian = phi_x.inverse_jacobian(qx);

      for (auto d = 0u; d < dim_x; d++)
        {
          auto new_val = jacobian[0][d] * grad_in[0];
          for (auto e = 1u; e < dim_x; ++e)
            new_val += (jacobian[e][d] * grad_in[e]);
          data_ptr_out[q + d * n_q_points_x * n_q_points_v] = new_val * jxw;
        }
    }



    /**
     * Submit gradient in v-space
     */
    inline DEAL_II_ALWAYS_INLINE //
      void
      submit_gradient_v(VectorizedArrayType *__restrict data_ptr_out,
                        const VectorizedArrayType *__restrict grad_in,
                        const unsigned int q,
                        const unsigned int qx,
                        const unsigned int qv) const
    {
      const auto jxw =
        phi_x.JxW(qx) * phi_v.JxW(qv)[n_vectors_v == 1 ? 0 : this->lane_y];
      const auto jacobian = phi_v.inverse_jacobian(qv);

      for (auto d = 0u; d < dim_v; d++)
        {
          auto new_val =
            jacobian[0][d][n_vectors_v == 1 ? 0 : this->lane_y] * grad_in[0];
          for (auto e = 1u; e < dim_v; ++e)
            new_val += (jacobian[e][d][n_vectors_v == 1 ? 0 : this->lane_y] *
                        grad_in[e]);
          data_ptr_out[q + d * n_q_points_x * n_q_points_v] = new_val * jxw;
        }
    }



    /**
     * Return location of quadrature point.
     *
     * TODO: split up in x- and v-space.
     */
    inline dealii::Point<dim, VectorizedArrayType>
    get_quadrature_point(const unsigned int q) const
    {
      dealii::Point<dim, VectorizedArrayType> temp;

      auto t1 = this->phi_x.quadrature_point(q % n_q_points_x);
      for (int i = 0; i < dim_x; i++)
        temp[i] = t1[i];
      auto t2 = this->phi_v.quadrature_point(q / n_q_points_x);
      for (int i = 0; i < dim_v; i++)
        temp[i + dim_x] = t2[i][n_vectors_v == 1 ? 0 : this->lane_y];

      return temp;
    }

    inline dealii::Point<dim, VectorizedArrayType>
    get_quadrature_point(const unsigned int qx, const unsigned int qv) const
    {
      dealii::Point<dim, VectorizedArrayType> temp;

      auto t1 = this->phi_x.quadrature_point(qx);
      for (int i = 0; i < dim_x; i++)
        temp[i] = t1[i];
      auto t2 = this->phi_v.quadrature_point(qv);
      for (int i = 0; i < dim_v; i++)
        temp[i + dim_x] = t2[i][n_vectors_v == 1 ? 0 : this->lane_y];

      return temp;
    }

    inline dealii::Point<dim_v, VectorizedArrayType>
    get_quadrature_point_v(const unsigned int qv) const
    {
      dealii::Point<dim_v, VectorizedArrayType> temp;

      auto t2 = this->phi_v.quadrature_point(qv);
      for (int i = 0; i < dim_v; i++)
        temp[i] = t2[i][n_vectors_v == 1 ? 0 : this->lane_y];

      return temp;
    }

  protected:
    // clang-format off
    dealii::FEEvaluation<dim_x, degree, n_points, 1, Number, typename PARENT::VectorizedArrayTypeX> phi_x;
    dealii::FEEvaluation<dim_v, degree, n_points, 1, Number, typename PARENT::VectorizedArrayTypeV> phi_v;
    // clang-format on
  };

} // namespace hyperdeal

#endif
