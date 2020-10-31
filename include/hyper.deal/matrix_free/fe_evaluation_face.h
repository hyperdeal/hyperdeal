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

#ifndef HYPERDEAL_NDIM_FEEVALUATION_FACE
#define HYPERDEAL_NDIM_FEEVALUATION_FACE

#include <hyper.deal/base/config.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <hyper.deal/matrix_free/fe_evaluation_base.h>
#include <hyper.deal/matrix_free/read_write_operation.h>
#include <hyper.deal/matrix_free/vector_access_internal.h>

namespace hyperdeal
{
  /**
   * The class that provides all functions necessary to evaluate functions at
   * quadrature points and face integrations in phase space. It delegates the
   * the actual evaluation task to two dealii::FEEvaluation and two
   * dealii::FEFaceEvaluation objects and combines the result on-the-fly.
   */
  template <int dim_x,
            int dim_v,
            int degree,
            int n_points,
            typename Number,
            typename VectorizedArrayType>
  class FEFaceEvaluation : public FEEvaluationBase<dim_x,
                                                   dim_v,
                                                   degree,
                                                   n_points,
                                                   Number,
                                                   VectorizedArrayType>
  {
  public:
    static const unsigned int DIM  = dim_x + dim_v;
    static const unsigned int DIM_ = DIM;
    static const unsigned int N_Q_POINTS =
      dealii::Utilities::pow(n_points, DIM - 1);

    static const unsigned int N_Q_POINTS_1_CELL =
      dealii::Utilities::pow(n_points, dim_x - 0);
    static const unsigned int N_Q_POINTS_2_CELL =
      dealii::Utilities::pow(n_points, dim_v - 0);

    static const unsigned int N_Q_POINTS_1_FACE =
      dealii::Utilities::pow(n_points, dim_x - 1);
    static const unsigned int N_Q_POINTS_2_FACE =
      dealii::Utilities::pow(n_points, dim_v - 1);

    using PARENT = FEEvaluationBase<dim_x,
                                    dim_v,
                                    degree,
                                    n_points,
                                    Number,
                                    VectorizedArrayType>;

    static const unsigned int n_vectors   = PARENT::n_vectors;
    static const unsigned int n_vectors_v = PARENT::n_vectors_v;

    // clang-format off
    static const unsigned int dim                    = dim_x + dim_v;
    static const unsigned int static_dofs_per_cell   = dealii::Utilities::pow(degree + 1, dim);
    static const unsigned int static_dofs_per_cell_x = dealii::Utilities::pow(degree + 1, dim_x);
    static const unsigned int static_dofs_per_cell_v = dealii::Utilities::pow(degree + 1, dim_v);
    static const unsigned int static_dofs_per_face   = dealii::Utilities::pow(degree + 1, dim_x + dim_v - 1);
    // clang-format on

    using SpaceType =
      typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::ID::
        SpaceType;

    /**
     * Constructor.
     *
     * @param matrix_free   Data object that contains all data.
     * @param is_minus_face This selects which of the two cells of an internal
     *                      face the current evaluator will be based upon.
     *                      The interior face is the main face along which the
     *                      normal vectors are oriented. The exterior face
     *                      coming from the other side provides the same normal
     *                      vector as the interior side, so if the outer normal
     *                      vector to that side is desired, it must be
     *                      multiplied by -1..
     * @param dof_no_x      If x-space matrix_free of matrix_free was set up
     *                      with multiple DoFHandler objects, this parameter
     *                      selects to which DoFHandler/AffineConstraints pair
     *                      the given evaluator should be attached to.
     * @param dof_no_v      Same as above but for v-space.
     * @param quad_no_x     If x-space matrix_free of matrix_free was set up
     *                      with multiple Quadrature objects, this parameter
     *                      selects the appropriate number of the quadrature
     *                      formula.
     * @param quad_no_v     Same as above but for v-space.
     */
    FEFaceEvaluation(
      const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &matrix_free,
      const bool         is_minus_face,
      const unsigned int dof_no_x,
      const unsigned int dof_no_v,
      const unsigned int quad_no_x,
      const unsigned int quad_no_v)
      : PARENT(matrix_free)
      , is_minus_face(is_minus_face)
      , phi_x(this->matrix_free_x, dof_no_x, quad_no_x)
      , phi_v(this->matrix_free_v, dof_no_v, quad_no_v)
      , phi_face_x(this->matrix_free_x, is_minus_face, dof_no_x, quad_no_x)
      , phi_face_v(this->matrix_free_v, is_minus_face, dof_no_v, quad_no_v)
    {
      this->shape_values = &phi_x.get_shape_info().data[0].shape_values_eo;
      this->shape_gradients =
        &phi_x.get_shape_info().data[0].shape_gradients_collocation_eo;
    }



    /**
     * Set the view the current face in the context of FCL.
     */
    void
    reinit(typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::ID
             face_index)
    {
      PARENT::reinit(face_index);
      this->type   = face_index.type;
      this->is_ecl = false;

      if (this->type == SpaceType::X)
        {
          phi_face_x.reinit(this->macro_cell_x);
          phi_v.reinit(this->macro_cell_v);
        }
      else
        {
          phi_x.reinit(this->macro_cell_x);
          phi_face_v.reinit(this->macro_cell_v);
        }

      if (this->type == SpaceType::X)
        {
          this->n_filled_lanes =
            this->matrix_free_x.n_active_entries_per_face_batch(
              this->macro_cell_x);
        }
      else
        {
          this->n_filled_lanes =
            this->matrix_free_x.n_active_entries_per_cell_batch(
              this->macro_cell_x);
        }

      // get direction of face
      if (type == SpaceType::X)
        {
          const dealii::internal::MatrixFreeFunctions::FaceToCellTopology<
            n_vectors> &faces =
            this->matrix_free_x.get_face_info(this->macro_cell_x);
          this->face_no = this->is_minus_face ? faces.interior_face_no :
                                                faces.exterior_face_no;
        }
      else
        {
          const dealii::internal::MatrixFreeFunctions::FaceToCellTopology<
            n_vectors_v> &faces =
            this->matrix_free_v.get_face_info(this->macro_cell_v);
          this->face_no = this->is_minus_face ? faces.interior_face_no :
                                                faces.exterior_face_no;
          this->face_no += dim_x * 2;
        }
    }



    /**
     * Set the view the current face in the context of ECL.
     */
    void
    reinit(typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::ID
                        cell_index,
           unsigned int face)
    {
      PARENT::reinit(cell_index);
      this->type   = (face >= 2 * dim_x) ? SpaceType::V : SpaceType::X;
      this->is_ecl = true;

      // delegate reinit to deal.II data structures
      if (this->type == SpaceType::X)
        {
          phi_face_x.reinit(this->macro_cell_x, face);
          phi_v.reinit(this->macro_cell_v);
        }
      else
        {
          phi_x.reinit(this->macro_cell_x);
          phi_face_v.reinit(this->macro_cell_v, face - 2 * dim_x);
        }

      // how many lanes are filled?
      {
        this->n_filled_lanes =
          this->matrix_free_x.n_active_entries_per_cell_batch(
            this->macro_cell_x);
      }

      // get direction of face
      {
        this->face_no = face;
      }
    }



    /**
     * Read data from a global vector into the internal buffer.
     */
    void
    read_dof_values(const dealii::LinearAlgebra::SharedMPI::Vector<Number> &src)
    {
      read_dof_values(src, &this->data[0]);
    }



    /**
     * Read data from a global vector into a given buffer.
     */
    void
    read_dof_values(const dealii::LinearAlgebra::SharedMPI::Vector<Number> &src,
                    VectorizedArrayType *data) const
    {
      if (this->matrix_free.are_ghost_faces_supported())
        {
          // for comments see dealii::FEEvaluation::reinit
          const unsigned int face_orientation =
            is_ecl ? 0 :
                     ((this->is_minus_face ==
                       (this->matrix_free.get_face_info()
                          .face_orientations[0][this->macro] >= 8)) ?
                        (this->matrix_free.get_face_info()
                           .face_orientations[0][this->macro] %
                         8) :
                        0);

          internal::MatrixFreeFunctions::ReadWriteOperation<Number>(
            this->matrix_free.get_dof_info(),
            this->matrix_free.get_face_info(),
            this->matrix_free.get_shape_info())
            .template process_face<dim_x, dim_v, degree>(
              internal::MatrixFreeFunctions::
                VectorReader<Number, VectorizedArrayType>(),
              src.shared_vector_data(),
              data,
              (is_ecl == false || this->is_minus_face) ?
                &this->face_no :
                &this->matrix_free.get_face_info()
                   .no_faces[3][(2 * dim * this->macro + this->face_no) *
                                n_vectors],
              (is_ecl == false || this->is_minus_face) ?
                &face_orientation :
                &this->matrix_free.get_face_info().face_orientations
                   [3][(2 * dim * this->macro + this->face_no) * n_vectors],
              this->type == SpaceType::X ? 0 : 8,
              this->macro,
              is_ecl ? 2 : !is_minus_face,
              (is_ecl == false || this->is_minus_face) ?
                this->macro :
                2 * dim * this->macro + this->face_no,
              !is_minus_face + (is_ecl ? 2 : 0));
        }
      else
        {
          AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
        }
    }



    /*
     *  Read the face data from a cell buffer held e.g. by FEEvaluation.
     */
    void
    read_dof_values_from_buffer(const VectorizedArrayType *src)
    {
      const auto &index_array = this->matrix_free.get_shape_info()
                                  .face_to_cell_index_nodal[this->face_no];

      for (unsigned int i = 0; i < static_dofs_per_face; ++i)
        this->data[i] = src[index_array[i]];
    }



    /**
     *  Add the face data into a cell buffer held e.g. by FEEvaluation.
     */
    void
    distribute_to_buffer(VectorizedArrayType *dst) const
    {
      const auto &index_array = this->matrix_free.get_shape_info()
                                  .face_to_cell_index_nodal[this->face_no];

      for (unsigned int i = 0; i < static_dofs_per_face; ++i)
        dst[index_array[i]] += this->data[i];
    }


    /**
     * Add the face data into a global vector.
     */
    void
    distribute_local_to_global(
      dealii::LinearAlgebra::SharedMPI::Vector<Number> &dst) const
    {
      Assert(is_ecl == false, dealii::StandardExceptions::ExcNotImplemented());

      if (this->matrix_free.are_ghost_faces_supported())
        {
          // for comments see dealii::FEEvaluation::reinit
          const unsigned int face_orientation =
            is_ecl ? 0 :
                     ((this->is_minus_face ==
                       (this->matrix_free.get_face_info()
                          .face_orientations[0][this->macro] >= 8)) ?
                        (this->matrix_free.get_face_info()
                           .face_orientations[0][this->macro] %
                         8) :
                        0);

          internal::MatrixFreeFunctions::ReadWriteOperation<Number>(
            this->matrix_free.get_dof_info(),
            this->matrix_free.get_face_info(),
            this->matrix_free.get_shape_info())
            .template process_face<dim_x, dim_v, degree>(
              internal::MatrixFreeFunctions::
                VectorDistributorLocalToGlobal<Number, VectorizedArrayType>(),
              dst.shared_vector_data(),
              &this->data[0],
              &this->face_no,
              &face_orientation,
              this->type == SpaceType::X ? 0 : 8,
              this->macro,
              !is_minus_face,
              this->macro,
              !is_minus_face);
        }
      else
        {
          AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
        }
    }



    /**
     * Return number of filled lanes.
     */
    inline unsigned int
    n_vectorization_lanes_filled() const
    {
      return n_filled_lanes;
    }


    /**
     * Submit value (i.e. multiply by JxW).
     */
    template <SpaceType stype>
    inline DEAL_II_ALWAYS_INLINE //
      void
      submit_value(VectorizedArrayType *__restrict data_ptr,
                   const VectorizedArrayType &value,
                   const unsigned int         q,
                   const unsigned int         qx,
                   const unsigned int         qv) const
    {
      Assert(stype == this->type, dealii::ExcMessage("Types do not match!"));

      // note: this if-statement is evaluated at compile time
      if (stype == SpaceType::X)
        data_ptr[q] = -value * phi_face_x.JxW(qx) *
                      phi_v.JxW(qv)[n_vectors_v == 1 ? 0 : this->lane_y];
      else if (stype == SpaceType::V)
        data_ptr[q] = -value * phi_x.JxW(qx) *
                      phi_face_v.JxW(qv)[n_vectors_v == 1 ? 0 : this->lane_y];
    }



    /**
     * The same as above but tested from both sides -> FCL.
     */
    template <SpaceType stype>
    inline DEAL_II_ALWAYS_INLINE //
      void
      submit_value(VectorizedArrayType *__restrict data_ptr_1,
                   VectorizedArrayType *__restrict data_ptr_2,
                   const VectorizedArrayType &value,
                   const unsigned int         q,
                   const unsigned int         q1,
                   const unsigned int         q2) const
    {
      Assert(stype == this->type, dealii::ExcMessage("Types do not match!"));

      // note: this if-statement is evaluated at compile time
      const auto temp =
        (stype == SpaceType::X) ?
          (value * phi_face_x.JxW(q1) *
           phi_v.JxW(q2)[n_vectors_v == 1 ? 0 : this->lane_y]) :
          (value * phi_x.JxW(q1) *
           phi_face_v.JxW(q2)[n_vectors_v == 1 ? 0 : this->lane_y]);
      data_ptr_1[q] = -temp;
      data_ptr_2[q] = +temp;
    }



    /**
     * Get normal for an x-face
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<1, dim_x, VectorizedArrayType>
      get_normal_vector_x(const unsigned int qx) const
    {
      // TODO: assert that we have (face x cell)

      return phi_face_x.get_normal_vector(qx);
    }



    /**
     * Get normal for a y-face
     */
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<1, dim_v, VectorizedArrayType>
      get_normal_vector_v(const unsigned int qv) const
    {
      // TODO: assert that we have (cell x face)

      // here we unfortunately have to copy the content due to different data
      // types!
      // TODO: implement constructor VectorizedArray<Number,
      // N>::VectorizedArray(VectorizedArray<Number, 1>)
      dealii::Tensor<1, dim_v, VectorizedArrayType> result;
      const auto temp = phi_face_v.get_normal_vector(qv);
      for (auto i = 0u; i < dim_v; i++)
        result[i] = temp[i][0];
      return result;
    }



    /**
     * Return position of quadrature point.
     *
     * TODO: specialize for x-space and v-space.
     */
    inline dealii::Point<DIM, VectorizedArrayType>
    get_quadrature_point(const unsigned int q) const
    {
      dealii::Point<DIM, VectorizedArrayType> temp;

      if (type == SpaceType::X)
        {
          const auto t1 =
            this->phi_face_x.quadrature_point(q % N_Q_POINTS_1_FACE);
          for (int i = 0; i < dim_x; i++)
            temp[i] = t1[i];
          const auto t2 = this->phi_v.quadrature_point(q / N_Q_POINTS_1_FACE);
          for (int i = 0; i < dim_v; i++)
            temp[i + dim_x] = t2[i][n_vectors_v == 1 ? 0 : this->lane_y];
        }
      else
        {
          const auto t1 = this->phi_x.quadrature_point(q % N_Q_POINTS_1_CELL);
          for (int i = 0; i < dim_x; i++)
            temp[i] = t1[i];
          const auto t2 =
            this->phi_face_v.quadrature_point(q / N_Q_POINTS_1_CELL);
          for (int i = 0; i < dim_v; i++)
            temp[i + dim_x] = t2[i][n_vectors_v == 1 ? 0 : this->lane_y];
        }

      return temp;
    }


    /**
     * Return coordinate of quadrature point (@p qx, @p qv).
     */
    template <SpaceType stype>
    inline dealii::Point<dim, VectorizedArrayType>
    get_quadrature_point(const unsigned int qx, const unsigned int qv) const
    {
      Assert(stype == this->type, dealii::ExcMessage("Types do not match!"));

      dealii::Point<dim, VectorizedArrayType> temp;

      if (stype == SpaceType::X)
        {
          const auto t1 = this->phi_face_x.quadrature_point(qx);
          for (int i = 0; i < dim_x; i++)
            temp[i] = t1[i];
          const auto t2 = this->phi_v.quadrature_point(qv);
          for (int i = 0; i < dim_v; i++)
            temp[i + dim_x] = t2[i][n_vectors_v == 1 ? 0 : this->lane_y];
        }
      else
        {
          const auto t1 = this->phi_x.quadrature_point(qx);
          for (int i = 0; i < dim_x; i++)
            temp[i] = t1[i];
          const auto t2 = this->phi_face_v.quadrature_point(qv);
          for (int i = 0; i < dim_v; i++)
            temp[i + dim_x] = t2[i][n_vectors_v == 1 ? 0 : this->lane_y];
        }

      return temp;
    }

  private:
    /**
     * Interior or exterior face.
     */
    const bool is_minus_face;

    /**
     * Number of filled lanes.
     */
    unsigned int n_filled_lanes;

    /**
     * Has reinit() been called within an ECL or FCL context.
     */
    bool is_ecl;

    /**
     * x- or v-space face.
     */
    SpaceType type;

    /**
     * Face number < dim * 2.
     */
    unsigned int face_no;

    // clang-format off
    dealii::FEEvaluation<dim_x, degree, n_points, 1, Number, typename PARENT::VectorizedArrayTypeX> phi_x;
    dealii::FEEvaluation<dim_v, degree, n_points, 1, Number, typename PARENT::VectorizedArrayTypeV> phi_v;

    dealii::FEFaceEvaluation<dim_x, degree, n_points, 1, Number, typename PARENT::VectorizedArrayTypeX> phi_face_x;
    dealii::FEFaceEvaluation<dim_v, degree, n_points, 1, Number, typename PARENT::VectorizedArrayTypeV> phi_face_v;
    // clang-format on
  };

} // namespace hyperdeal

#endif
