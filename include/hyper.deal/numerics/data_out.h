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

#ifndef NDIM_DATA_OUT
#define NDIM_DATA_OUT

#include <hyper.deal/base/config.h>

#include <deal.II/numerics/data_out.h>

#include <hyper.deal/matrix_free/fe_evaluation_cell.h>
#include <hyper.deal/matrix_free/matrix_free.h>

namespace hyperdeal
{
  namespace DataOut
  {
    template <int degree,
              int n_points,
              int dim_x,
              int dim_v,
              typename Number,
              typename VectorType,
              typename VectorizedArrayType>
    void
    write_vtu_in_parallel(
      const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &matrix_free,
      const VectorType &                                           src,
      const unsigned int                                           dof_no_x,
      const unsigned int                                           dof_no_v,
      const unsigned int                                           quad_no_x,
      const unsigned int                                           quad_no_v,
      const unsigned int time_step_counter)
    {
      (void)matrix_free;
      (void)src;
      (void)dof_no_x;
      (void)dof_no_v;
      (void)quad_no_x;
      (void)quad_no_v;
      (void)time_step_counter;

      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
    }

    template <int degree,
              int n_points,
              typename Number,
              typename VectorType,
              typename VectorizedArrayType>
    void
    write_vtu_in_parallel(
      const MatrixFree<1, 1, Number, VectorizedArrayType> &matrix_free,
      const VectorType &                                   src,
      const unsigned int                                   dof_no_x,
      const unsigned int                                   dof_no_v,
      const unsigned int                                   quad_no_x,
      const unsigned int                                   quad_no_v,
      const unsigned int                                   time_step_counter)
    {
      const static int dim_x = 1;
      const static int dim_v = 1;
      const static int dim   = dim_x + dim_v;

      const auto n_cells = matrix_free.get_matrix_free_x().n_physical_cells() *
                           matrix_free.get_matrix_free_v().n_physical_cells();

      FEEvaluation<dim_x, dim_v, degree, n_points, Number, VectorizedArrayType>
        phi(matrix_free, dof_no_x, dof_no_v, quad_no_x, quad_no_v);

      int dummy;

      std::vector<dealii::Point<dim>>    points(n_cells * 4);
      std::vector<dealii::CellData<dim>> cells(n_cells);
      dealii::Vector<Number> solution(n_cells * n_points * n_points);

      unsigned int cell_counter = 0;

      matrix_free.template cell_loop<int, VectorType>(
        [&](
          const auto &, int &, const VectorType &src, const auto cell) mutable {
          phi.reinit(cell);
          phi.read_dof_values(src);

          const VectorizedArrayType *data_ptr_src = phi.get_data_ptr();

          const auto point_0 = phi.get_quadrature_point(0, 0);
          const auto point_1 = phi.get_quadrature_point(degree, degree);

          for (unsigned int v = 0; v < phi.n_vectorization_lanes_filled();
               v++, cell_counter++)
            {
              for (unsigned int i = 0; i < 4; ++i)
                cells[cell_counter].vertices[i] = cell_counter * 4 + i;

              points[cell_counter * 4 + 0] =
                dealii::Point<dim>(point_0[0][v], point_0[1][v]);
              points[cell_counter * 4 + 1] =
                dealii::Point<dim>(point_1[0][v], point_0[1][v]);
              points[cell_counter * 4 + 2] =
                dealii::Point<dim>(point_0[0][v], point_1[1][v]);
              points[cell_counter * 4 + 3] =
                dealii::Point<dim>(point_1[0][v], point_1[1][v]);

              for (unsigned int qv = 0, q = 0; qv < phi.n_q_points_v; qv++)
                for (unsigned int qx = 0; qx < phi.n_q_points_x; qx++, q++)
                  solution[cell_counter * phi.n_q_points_x * phi.n_q_points_v +
                           q] = data_ptr_src[q][v];
            }
        },
        dummy,
        src);

      dealii::Triangulation<dim> tria;
      tria.create_triangulation(points, cells, {});

      dealii::DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(dealii::FE_DGQ<dim>(degree));

      dealii::DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      dealii::DataOut<dim> data_out;
      data_out.set_flags(flags);

      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");
      data_out.build_patches(degree);
      const std::string filename =
        "solution_tp_" + std::to_string(time_step_counter) + ".vtu";
      data_out.write_vtu_in_parallel(filename, matrix_free.get_communicator());
    }

  } // namespace DataOut

} // namespace hyperdeal

#endif
