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

#ifndef HYPERDEAL_CFL
#define HYPERDEAL_CFL

#include <hyper.deal/base/config.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

namespace hyperdeal
{
  namespace advection
  {
    /**
     * Return critical time step.
     */
    template <int dim, typename Number>
    Number
    compute_critical_time_step(dealii::DoFHandler<dim> &      dof_handler,
                               dealii::Tensor<1, dim, Number> u,
                               const unsigned int             degree,
                               const unsigned int             n_points,
                               MPI_Comm                       comm);

    /**
     * Return critical time step phase space.
     */
    template <int dim_x, int dim_v, typename Number>
    Number
    compute_critical_time_step(dealii::DoFHandler<dim_x> &dof_handler_x,
                               dealii::DoFHandler<dim_v> &dof_handler_v,
                               dealii::Tensor<1, dim_x + dim_v, Number> uu,
                               const unsigned int degree_x,
                               const unsigned int n_points_x,
                               MPI_Comm           comm_row,
                               const unsigned int degree_v,
                               const unsigned int n_points_v,
                               MPI_Comm           comm_column);



    template <int dim, typename Number>
    Number
    compute_critical_time_step(dealii::DoFHandler<dim> &      dof_handler,
                               dealii::Tensor<1, dim, Number> u,
                               const unsigned int             degree,
                               const unsigned int             n_points,
                               MPI_Comm                       comm)
    {
      dealii::FE_DGQ<dim> fe(degree);
      dealii::QGauss<dim> quad(n_points);

      dealii::FEValues<dim> fe_values(fe,
                                      quad,
                                      dealii::update_inverse_jacobians);

      Number v_max = 0.0;

      for (auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            for (unsigned int q = 0; q < quad.size(); ++q)
              {
                dealii::Tensor<1, dim, Number> v;
                for (unsigned int i = 0; i < dim; i++)
                  for (unsigned int j = 0; j < dim; j++)
                    v[j] += fe_values.inverse_jacobian(q)[i][j] * u[i];

                for (unsigned int i = 0; i < dim; i++)
                  v_max = std::max(v_max, std::abs(v[i]));
              }
          }

      return 1.0 / dealii::Utilities::MPI::max(v_max, comm);
    }



    template <int dim_x, int dim_v, typename Number>
    Number
    compute_critical_time_step(dealii::DoFHandler<dim_x> &dof_handler_x,
                               dealii::DoFHandler<dim_v> &dof_handler_v,
                               dealii::Tensor<1, dim_x + dim_v, Number> uu,
                               const unsigned int degree_x,
                               const unsigned int n_points_x,
                               MPI_Comm           comm_row,
                               const unsigned int degree_v,
                               const unsigned int n_points_v,
                               MPI_Comm           comm_column)
    {
      Number v1, v2;
      {
        dealii::Tensor<1, dim_x, Number> u;
        for (unsigned int i = 0; i < dim_x; i++)
          u[i] = uu[i];
        v1 = compute_critical_time_step(
          dof_handler_x, u, degree_x, n_points_x, comm_row);
      }

      {
        dealii::Tensor<1, dim_v, Number> u;
        for (unsigned int i = 0; i < dim_v; i++)
          u[i] = uu[i + dim_x];
        v2 = compute_critical_time_step(
          dof_handler_v, u, degree_v, n_points_v, comm_column);
      }

      return std::min(v1, v2);
    }

  } // namespace advection
} // namespace hyperdeal

#endif
