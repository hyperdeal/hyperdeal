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

#ifndef HYPERDEAL_DERIVATIVE_CONTAINER
#define HYPERDEAL_DERIVATIVE_CONTAINER

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/tensor.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace hyperdeal
{
  namespace vp
  {
    /**
     * Data structure storing some information at cell/face quadrature points
     * (stored such a way as matrix-free needs it).
     *
     * TODO: add support for ECL for faces. Currently not needed for VP.
     */
    template <int dim, int n_points, typename UnitType>
    class QuadraturePointContainer
    {
      static const unsigned int n_points_cell = std::pow(n_points, dim - 0);
      static const unsigned int n_points_face = std::pow(n_points, dim - 1);

    public:
      /**
       * Return the value at a cell quadrature point.
       */
      inline DEAL_II_ALWAYS_INLINE //
        UnitType
        get_value_cell(const unsigned int cell_index,
                       const unsigned int q_index) const
      {
        return values_cells[cell_index * n_points_cell + q_index];
      }

      /**
       * Return the value at a interior finner face quadrature point (FCL).
       */
      inline DEAL_II_ALWAYS_INLINE //
        UnitType
        get_value_face_interior(const unsigned int face_index,
                                const unsigned int q_index) const
      {
        AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());

        return values_faces_int[face_index * n_points_face + q_index];
      }

      /**
       * Return the value at a exterior finner ace quadrature point (FCL).
       */
      inline DEAL_II_ALWAYS_INLINE //
        UnitType
        get_value_face_exterior(const unsigned int face_index,
                                const unsigned int q_index) const
      {
        AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());

        return values_faces_ext[face_index * n_points_face + q_index];
      }

      virtual std::size_t
      memory_consumption() const
      {
        return values_cells.memory_consumption() +
               values_faces_int.memory_consumption() +
               values_faces_ext.memory_consumption();
      }
      /**
       * Allocate memory
       */
      template <typename MF>
      void
      reinit(const MF &data)
      {
        if (values_cells.size() != data.n_cell_batches() * n_points_cell)
          values_cells.resize(data.n_cell_batches() * n_points_cell);

#ifdef false
        if (values_faces_int.size() !=
            (data.n_inner_face_batches() + data.n_boundary_face_batches()) *
              n_points_face)
          values_faces_int.resize(
            (data.n_inner_face_batches() + data.n_boundary_face_batches()) *
            n_points_face);

        if (values_faces_ext.size() !=
            data.n_inner_face_batches() * n_points_face)
          values_faces_ext.resize(data.n_inner_face_batches() * n_points_face);
#endif
      }


    protected:
      /**
       * Values at cell quadrature points.
       */
      dealii::AlignedVector<UnitType> values_cells;

      /**
       * Values at interior internal face quadrature points.
       */
      dealii::AlignedVector<UnitType> values_faces_int;

      /**
       * Values at exterior internal face quadrature points.
       */
      dealii::AlignedVector<UnitType> values_faces_ext;
    };

    /**
     * Data structure storing the precomputed derivative of a solution field
     * at the quadrature points.
     *
     * @note This class does nothing else than computing the derivatives. The
     *   actual storage of the values and the access to them is handled by
     *   the class QuadraturePointContainer.
     */
    template <int dim,
              int degree,
              int n_points,
              typename Number,
              typename VectorizedArrayType>
    class DerivativeContainer : public QuadraturePointContainer<
                                  dim,
                                  n_points,
                                  dealii::Tensor<1, dim, VectorizedArrayType>>
    {
      static const unsigned int n_points_cell = std::pow(n_points, dim - 0);
      static const unsigned int n_points_face = std::pow(n_points, dim - 1);

    public:
      /*
       * Recompute derivatives.
       */
      template <typename VectorType>
      void
      update(
        const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
        const VectorType &                                          src)
      {
        this->reinit(matrix_free);

        dealii::
          FEEvaluation<dim, degree, n_points, 1, Number, VectorizedArrayType>
            phi_rho(matrix_free);
        dealii::FEFaceEvaluation<dim,
                                 degree,
                                 n_points,
                                 1,
                                 Number,
                                 VectorizedArrayType>
          phi_rho_i(matrix_free, true);
        dealii::FEFaceEvaluation<dim,
                                 degree,
                                 n_points,
                                 1,
                                 Number,
                                 VectorizedArrayType>
          phi_rho_e(matrix_free, false);

        int dummy;

        matrix_free.template loop<int, VectorType>(
          [&](const auto &, auto &, const auto &src, const auto range) {
            // cells
            for (auto cell = range.first; cell < range.second; ++cell)
              {
                phi_rho.reinit(cell);
                phi_rho.gather_evaluate(src,
                                        dealii::EvaluationFlags::gradients);
                for (unsigned int q = 0; q < n_points_cell; ++q)
                  this->values_cells[cell * n_points_cell + q] =
                    phi_rho.get_gradient(q);
              }
          },
          [&](const auto &, auto &, const auto &src, const auto range) {
        // inner faces

#if true
            (void)src;
            (void)range;
#else
            for (auto face = range.first; face < range.second; ++face)
              {
                // ... interior face
                phi_rho_i.reinit(face);
                phi_rho_i.gather_evaluate(src,
                                          dealii::EvaluationFlags::gradients);
                for (unsigned int q = 0; q < n_points_face; ++q)
                  this->values_faces_int[face * n_points_face + q] =
                    phi_rho_i.get_gradient(q);

                // ... exterior face
                phi_rho_e.reinit(face);
                phi_rho_e.gather_evaluate(src,
                                          dealii::EvaluationFlags::gradients);
                for (unsigned int q = 0; q < n_points_face; ++q)
                  this->values_faces_ext[face * n_points_face + q] =
                    phi_rho_e.get_gradient(q);
              }
#endif
          },
          [&](const auto &, auto &, const auto &src, const auto range) {
        // boundary faces

#if true
            (void)src;
            (void)range;
#else
            for (auto face = range.first; face < range.second; ++face)
              {
                phi_rho_i.reinit(face);
                phi_rho_i.gather_evaluate(src,
                                          dealii::EvaluationFlags::gradients);
                for (unsigned int q = 0; q < n_points_face; ++q)
                  this->values_faces_int[face * n_points_face + q] =
                    phi_rho_i.get_gradient(q);
              }
#endif
          },
          dummy,
          src,
          false,
          dealii::MatrixFree<dim, Number, VectorizedArrayType>::
            DataAccessOnFaces::none,
          dealii::MatrixFree<dim, Number, VectorizedArrayType>::
            DataAccessOnFaces::gradients);
      }
    };

  } // namespace vp
} // namespace hyperdeal

#endif
