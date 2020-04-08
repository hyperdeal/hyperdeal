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

#ifndef NDIM_PHASE_SPACE_DIAGNOSTICS
#define NDIM_PHASE_SPACE_DIAGNOSTICS

#include <hyper.deal/matrix_free/fe_evaluation_cell.h>
#include <hyper.deal/matrix_free/matrix_free.h>
#include <hyper.deal/numerics/vector_tools.h>

namespace hyperdeal
{
  /**
   * [TODO]
   */
  template <int degree,
            int n_points,
            int dim_x,
            int dim_v,
            typename Number,
            typename VectorType,
            typename VectorizedArrayType>
  dealii::Tensor<1, 6, Number>
  phase_space_diagnostics(
    const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &matrix_free,
    const VectorType &                                           src)
  {
    FEEvaluation<dim_x, dim_v, degree, n_points, Number, VectorizedArrayType>
      phi(matrix_free, 0, 0, 0, 0);

    dealii::Tensor<1, 6, Number> result;

    int dummy;

    matrix_free.template cell_loop<int, VectorType>(
      [&](const auto &, int &, const VectorType &src, const auto cell) mutable {
        const VectorizedArrayType *f_ptr =
          VectorTools::internal::interpolate(phi, src, cell);

        dealii::Tensor<1, 6, VectorizedArrayType> temp;

        for (unsigned int qv = 0, q = 0; qv < phi.n_q_points_v; qv++)
          {
            const auto v   = phi.get_quadrature_point_v(qv);
            const auto vxv = v * v;

            for (unsigned int qx = 0; qx < phi.n_q_points_x; qx++, q++)
              {
                const auto f   = f_ptr[q];
                const auto JxW = phi.JxW(qx, qv);

                temp[0] += f * JxW;       // mass
                temp[1] += f * f * JxW;   // l2 norm
                temp[2] += vxv * f * JxW; // kinetic energy

                // components of the momentum
                for (unsigned int d = 0; d < dim_v; ++d)
                  temp[3 + d] += v[d] * f * JxW;
              }
          }

        // gather results (VectorizedArray<Number> -> Number)
        for (unsigned int v = 0; v < phi.n_vectorization_lanes_filled(); v++)
          for (unsigned int i = 0; i < 6; i++)
            result[i] += temp[i][v];
      },
      dummy,
      src);

    result =
      dealii::Utilities::MPI::sum(result, matrix_free.get_communicator());
    result[1] = std::sqrt(result[1]);

    return result;
  }

  /**
   * [TODO]
   */
  template <int degree,
            int n_points,
            int dim_x,
            typename Number,
            typename VectorType,
            typename VectorizedArrayType>
  dealii::Tensor<1, dim_x, Number>
  compute_electric_energy(
    const dealii::MatrixFree<dim_x, Number, VectorizedArrayType> &data,
    const VectorType &                                            src)
  {
    dealii::Tensor<1, dim_x, Number> accumulated_electric_energy;

    dealii::
      FEEvaluation<dim_x, degree, n_points, 1, Number, VectorizedArrayType>
        phi_rho(data);

    int dummy;

    data.template cell_loop<int, VectorType>(
      [&](
        const auto &, int &, const VectorType &src, const auto range) mutable {
        for (unsigned int cell = range.first; cell < range.second; ++cell)
          {
            dealii::Tensor<1, dim_x, VectorizedArrayType> local_sum;
            phi_rho.reinit(cell);
            phi_rho.gather_evaluate(src, false, true);
            for (unsigned int q = 0; q < phi_rho.n_q_points; ++q)
              for (unsigned int d = 0; d < dim_x; ++d)
                local_sum[d] += phi_rho.get_gradient(q)[d] *
                                phi_rho.get_gradient(q)[d] * phi_rho.JxW(q);

            for (unsigned int v = 0;
                 v < data.n_active_entries_per_cell_batch(cell);
                 ++v)
              for (unsigned int d = 0; d < dim_x; ++d)
                accumulated_electric_energy[d] += local_sum[d][v];
          }
      },
      dummy,
      src);

    const auto tria =
      dynamic_cast<const dealii::parallel::TriangulationBase<dim_x> *>(
        &data.get_dof_handler().get_triangulation());

    const MPI_Comm comm = tria ? tria->get_communicator() : MPI_COMM_SELF;

    return dealii::Utilities::MPI::sum(accumulated_electric_energy, comm);
  }

} // namespace hyperdeal

#endif
