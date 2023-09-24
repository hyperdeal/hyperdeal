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

#ifndef NDIM_INTERPOLATION
#define NDIM_INTERPOLATION

#include <hyper.deal/base/config.h>

#include <deal.II/base/function.h>

#include <hyper.deal/matrix_free/fe_evaluation_cell.h>
#include <hyper.deal/matrix_free/matrix_free.h>

namespace hyperdeal
{
  namespace VectorTools
  {
    namespace internal
    {
      template <int dim_x,
                int dim_v,
                int degree,
                int n_points,
                typename Number,
                typename VectorizedArrayType,
                typename VectorType,
                typename ID>
      VectorizedArrayType *
      interpolate(FEEvaluation<dim_x,
                               dim_v,
                               degree,
                               n_points,
                               Number,
                               VectorizedArrayType> &phi,
                  const VectorType &                 src,
                  const ID                           cell)
      {
        static const int dim = dim_x + dim_v;

        static const dealii::internal::EvaluatorVariant tensorproduct =
          dealii::internal::EvaluatorVariant::evaluate_evenodd;


        // get data and scratch
        VectorizedArrayType *data_ptr = phi.get_data_ptr();

        // get cell values
        phi.reinit(cell);
        phi.read_dof_values(src);

        dealii::internal::FEEvaluationImplBasisChange<
          tensorproduct,
          dealii::internal::EvaluatorQuantity::value,
          dim,
          degree + 1,
          n_points>::do_forward(1, *phi.get_shape_values(), data_ptr, data_ptr);

        return data_ptr;
      }
    } // namespace internal

    /**
     * Compute the interpolation of the function @p analytical_solution at the
     * support points to the finite element space described by the DoFHandlers
     * (@p dof_no_x and @p dof_no_v) and the quadrature rule (@p quad_no_x and
     * @p quad_no_v), which should correspond to the FiniteElement object in
     * use.
     */
    template <int degree,
              int n_points,
              int dim_x,
              int dim_v,
              typename Number,
              typename VectorType,
              typename VectorizedArrayType>
    void
    interpolate(
      const std::shared_ptr<dealii::Function<dim_x + dim_v, Number>>
                                                                   analytical_solution,
      const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &matrix_free,
      VectorType &                                                 dst,
      const unsigned int                                           dof_no_x,
      const unsigned int                                           dof_no_v,
      const unsigned int                                           quad_no_x,
      const unsigned int                                           quad_no_v)
    {
      const static int dim = dim_x + dim_v;

      FEEvaluation<dim_x, dim_v, degree, n_points, Number, VectorizedArrayType>
        phi(matrix_free, dof_no_x, dof_no_v, quad_no_x, quad_no_v);

      const int dummy = 0;

      matrix_free.template cell_loop<VectorType, int>(
        [&](const auto &, auto &dst, const auto &, const auto cell) mutable {
          VectorizedArrayType *data_ptr = phi.get_data_ptr();

          phi.reinit(cell);

          // loop over quadrature points
          for (unsigned int qv = 0, q = 0; qv < phi.n_q_points_v; qv++)
            for (unsigned int qx = 0; qx < phi.n_q_points_x; qx++, q++)
              {
                // get reference to quadrature points
                const auto q_point = phi.get_quadrature_point(qx, qv);
                // loop over all lanes
                for (unsigned int v = 0; v < phi.n_vectorization_lanes_filled();
                     v++)
                  {
                    // setup point ...
                    dealii::Point<dim, Number> p;
                    for (unsigned int d = 0; d < dim; ++d)
                      p[d] = q_point[d][v];
                    // ... evaluate function at point
                    data_ptr[q][v] = analytical_solution->value(p);
                  }
              }

          phi.set_dof_values(dst);
        },
        dst,
        dummy);
    }

    /**
     * Compute L2-norm and L2-norm of error for a given vector @p src and
     * solution @p analytical_solution. The DoFHandlers to be used can be
     * specified by @p dof_no_x and @p dof_no_v and the point in which the
     * error is evaluated (coinciding with quadrature points) by @p quad_no_x
     * and @p quad_no_v.
     */
    template <int degree,
              int n_points,
              int dim_x,
              int dim_v,
              typename Number,
              typename VectorType,
              typename VectorizedArrayType>
    std::array<Number, 2>
    norm_and_error(
      const std::shared_ptr<dealii::Function<dim_x + dim_v, Number>>
                                                                   analytical_solution,
      const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &matrix_free,
      const VectorType &                                           src,
      const unsigned int                                           dof_no_x,
      const unsigned int                                           dof_no_v,
      const unsigned int                                           quad_no_x,
      const unsigned int                                           quad_no_v)
    {
      const static int dim = dim_x + dim_v;

      FEEvaluation<dim_x, dim_v, degree, n_points, Number, VectorizedArrayType>
        phi(matrix_free, dof_no_x, dof_no_v, quad_no_x, quad_no_v);

      std::array<Number, 2> result{{0.0, 0.0}};

      int dummy;

      matrix_free.template cell_loop<int, VectorType>(
        [&](
          const auto &, int &, const VectorType &src, const auto cell) mutable {
          const VectorizedArrayType *f_ptr =
            VectorTools::internal::interpolate(phi, src, cell);

          std::array<VectorizedArrayType, 2> temp;
          std::fill(temp.begin(), temp.end(), 0.);

          for (unsigned int qv = 0, q = 0; qv < phi.n_q_points_v; qv++)
            for (unsigned int qx = 0; qx < phi.n_q_points_x; qx++, q++)
              {
                // determine exact solution at quadrature point
                VectorizedArrayType solution_q;
                const auto          q_point = phi.get_quadrature_point(qx, qv);

                for (unsigned int v = 0; v < phi.n_vectorization_lanes_filled();
                     v++)
                  {
                    dealii::Point<dim> p;
                    for (unsigned int d = 0; d < dim; ++d)
                      p[d] = q_point[d][v];
                    solution_q[v] = analytical_solution->value(p);
                  }

                const auto f   = f_ptr[q];       // value at quadrature point
                const auto e   = f - solution_q; // error
                const auto JxW = phi.JxW(qx, qv);

                temp[0] += f * f * JxW; // L2: norm
                temp[1] += e * e * JxW; // L2: error
              }

          // gather results
          for (unsigned int v = 0; v < phi.n_vectorization_lanes_filled(); v++)
            for (unsigned int i = 0; i < temp.size(); i++)
              result[i] += temp[i][v];
        },
        dummy,
        src);

      MPI_Allreduce(MPI_IN_PLACE,
                    result.data(),
                    result.size(),
                    MPI_DOUBLE, // [TODO]: mpi_type_id
                    MPI_SUM,
                    matrix_free.get_communicator());

      result[0] = std::sqrt(result[0]);
      result[1] = std::sqrt(result[1]);

      return result;
    }

    /**
     * Perform integration along the v-space using the v-space quadrature rule
     * specified by @p quad_no_v and the DoFHandlers specified by @p dof_no_x
     * and @dof_no_v.
     */
    template <int degree,
              int n_points,
              int dim_x,
              int dim_v,
              typename Number,
              typename VectorizedArrayType,
              typename Vector_Out,
              typename Vector_In>
    void
    velocity_space_integration(
      const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &data,
      Vector_Out &                                                 dst,
      const Vector_In &                                            src,
      const unsigned int                                           dof_no_x,
      const unsigned int                                           dof_no_v,
      const unsigned int                                           quad_no_v)
    {
      using MF =
        hyperdeal::MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>;

      using VectorizedArrayTypeX = typename MF::VectorizedArrayTypeX;
      using VectorizedArrayTypeV = typename MF::VectorizedArrayTypeV;

      FEEvaluation<dim_x, dim_v, degree, n_points, Number, VectorizedArrayType>
      phi(const_cast<MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &>(
            data),
          dof_no_x,
          dof_no_v,
          0 /*dummy*/,
          0 /*dummy*/);

      dealii::
        FEEvaluation<dim_x, degree, n_points, 1, Number, VectorizedArrayTypeX>
          phi_x(data.get_matrix_free_x(), dof_no_x, 0 /*dummy*/);
      dealii::
        FEEvaluation<dim_v, degree, n_points, 1, Number, VectorizedArrayTypeV>
          phi_v(data.get_matrix_free_v(), dof_no_v, quad_no_v);

      dst = 0.0; // clear destination vector

      data.template cell_loop<Vector_Out, Vector_In>(
        [&](const auto &,
            Vector_Out &     dst,
            const Vector_In &src,
            const auto       cell) mutable {
          phi.reinit(cell);
          phi.read_dof_values(src);

          phi_x.reinit(cell.x);

          unsigned int index_v = cell.v / VectorizedArrayTypeV::size();
          unsigned int lane_v  = cell.v % VectorizedArrayTypeV::size();
          phi_v.reinit(index_v);

          // get data and scratch
          const VectorizedArrayType *data_ptr_src = phi.get_data_ptr();
          VectorizedArrayTypeX *     data_ptr_dst = phi_x.begin_dof_values();

          // loop over all x points and integrate over all v points
          for (unsigned int qx = 0; qx < phi_x.n_q_points; ++qx)
            {
              VectorizedArrayType sum_v = VectorizedArrayType();
              for (unsigned int qv = 0;
                   qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v);
                   ++qv)
                sum_v +=
                  data_ptr_src[qx +
                               dealii::Utilities::pow(n_points, dim_v) * qv] *
                  phi_v.JxW(qv)[lane_v];
              data_ptr_dst[qx] = sum_v;
            }

          phi_x.distribute_local_to_global(dst);
        },
        dst,
        src);

      // collect global contributions
      MPI_Allreduce(
        MPI_IN_PLACE,
        dst.get_values(),
        dst.locally_owned_size(),
        MPI_DOUBLE,
        MPI_SUM,
        data.get_matrix_free_v().get_dof_handler().get_communicator());
    }

    /**
     * Perform integration along the x-space using the x-space quadrature rule
     * specified by @p quad_no_x and DoFHandlers specified by @p dof_no_x and
     * @p dof_no_v.
     */
    template <int degree,
              int n_points,
              int dim_x,
              int dim_v,
              typename Number,
              typename VectorizedArrayType,
              typename Vector_Out,
              typename Vector_In>
    void
    coordinate_space_integration(
      const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &data,
      Vector_Out &                                                 dst,
      const Vector_In &                                            src,
      const unsigned int                                           dof_no_x,
      const unsigned int                                           dof_no_v,
      const unsigned int                                           quad_no_x)
    {
      using MF =
        hyperdeal::MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>;
      using VectorizedArrayTypeX = typename MF::VectorizedArrayTypeX;
      using VectorizedArrayTypeV = typename MF::VectorizedArrayTypeV;
      static_assert(VectorizedArrayTypeV::size() == 1);

      FEEvaluation<dim_x, dim_v, degree, n_points, Number, VectorizedArrayType>
        phi(data, dof_no_x, dof_no_v, 0 /*dummy*/, 0 /*dummy*/);
      dealii::
        FEEvaluation<dim_x, degree, n_points, 1, Number, VectorizedArrayTypeX>
          phi_x(data.get_matrix_free_x(), dof_no_x, quad_no_x);
      dealii::
        FEEvaluation<dim_v, degree, n_points, 1, Number, VectorizedArrayTypeV>
          phi_v(data.get_matrix_free_v(), dof_no_v, 0 /*dummy*/);

      // Clear destination vector
      dst = 0.0;

      constexpr auto N_v_points = dealii::Utilities::pow(n_points, dim_v);

      dealii::AlignedVector<VectorizedArrayTypeX> scratch(N_v_points);

      data.template cell_loop<Vector_Out, Vector_In>(
        [&](const auto &,
            Vector_Out &     dst,
            const Vector_In &src,
            const auto       cell) mutable {
          phi.reinit(cell);
          phi.read_dof_values(src);
          phi_x.reinit(cell.x);
          phi_v.reinit(cell.v);

          const VectorizedArrayType *data_ptr_src = phi.get_data_ptr();
          VectorizedArrayTypeV *     data_ptr_dst = phi_v.begin_dof_values();

          const auto N_lanes_x = phi.n_vectorization_lanes_filled();

          // reduce in x-direction in a vectorized fashion
          for (unsigned int qv = 0; qv < phi_v.n_q_points; ++qv)
            {
              scratch[qv] = 0.0;
              for (unsigned int qx = 0; qx < phi_x.n_q_points; ++qx)
                scratch[qv] +=
                  data_ptr_src[qx + qv * N_v_points] * phi_x.JxW(qx);
            }

          // perform cross-lane reduction
          for (unsigned int qv = 0; qv < phi_v.n_q_points; ++qv)
            {
              data_ptr_dst[qv] = 0.0;
              for (unsigned int lane = 0; lane < N_lanes_x; ++lane)
                data_ptr_dst[qv] += scratch[qv][lane];
            }

          phi_v.distribute_local_to_global(dst);
        },
        dst,
        src);

      // Collect global contribution
      const auto comm =
        data.get_matrix_free_x().get_dof_handler().get_communicator();
      MPI_Allreduce(MPI_IN_PLACE,
                    &*dst.begin(),
                    dst.locally_owned_size(),
                    MPI_DOUBLE,
                    MPI_SUM,
                    comm);
    }

  } // namespace VectorTools



} // namespace hyperdeal

#endif
