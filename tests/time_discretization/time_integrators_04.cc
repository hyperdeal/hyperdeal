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


// Test integration of a simple ODE: y' = y*sin(t)^2
// Its exact solution is y(t) = y0 * exp( (t - sin(t)*cos(t))/2 )

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>

#include <hyper.deal/base/time_integrators.h>
#include <hyper.deal/base/time_integrators.templates.h>

#include <tuple>

#include "../tests.h"

const auto comm = MPI_COMM_WORLD;

template <typename VectorType, bool is_block_vector>
std::tuple<VectorType, VectorType, VectorType>
construct_vectors()
{
  // Small values to get a meaningful partitioning
  constexpr int dim          = 2;
  constexpr int subdivisions = 4;
  constexpr int degree       = 1;

  // Set up a mock system
  parallel::distributed::Triangulation<dim> tria(comm);
  GridGenerator::subdivided_hyper_cube(tria, subdivisions, 0, 1);
  DoFHandler<dim> dof_handler(tria);

  if constexpr (is_block_vector)
    {
      constexpr auto      n_blocks = 2;
      const FESystem<dim> fe(FE_Q<dim>(degree), 1, FE_Q<dim>(degree), 1);

      dof_handler.distribute_dofs(fe);
      DoFRenumbering::block_wise(dof_handler);
      const auto dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler);
      const auto locally_owned_indices =
        dof_handler.locally_owned_dofs().split_by_block(dofs_per_block);
      const auto locally_relevant_indices =
        DoFTools::extract_locally_relevant_dofs(dof_handler)
          .split_by_block(dofs_per_block);

      VectorType vct_Ki(locally_owned_indices, comm);
      VectorType vct_Ti(locally_owned_indices, comm);
      VectorType vct_y(locally_owned_indices, comm);

      return std::make_tuple(vct_Ki, vct_Ti, vct_y);
    }
  else
    {
      const FE_Q<dim> fe(degree);

      dof_handler.distribute_dofs(fe);
      const auto locally_owned_indices = dof_handler.locally_owned_dofs();
      const auto locally_relevant_indices =
        DoFTools::extract_locally_relevant_dofs(dof_handler);

      VectorType vct_Ki(locally_owned_indices, comm);
      VectorType vct_Ti(locally_owned_indices, comm);
      VectorType vct_y(locally_owned_indices, comm);

      return std::make_tuple(vct_Ki, vct_Ti, vct_y);
    }
}

template <typename Number, typename VectorType, bool is_block_vector = false>
void
test()
{
  using Integrator =
    hyperdeal::LowStorageRungeKuttaIntegrator<Number, VectorType>;

  const auto   Niters = 100;
  const Number dt     = 0.1;
  const Number y0     = 1.0;

  auto [vct_Ki, vct_Ti, vct_y] =
    construct_vectors<VectorType, is_block_vector>();
  Integrator integrator(vct_Ki, vct_Ti, "rk45", /*only_Ti_is_ghosted=*/true);

  // Initial condition
  for (const auto i : vct_y.locally_owned_elements())
    vct_y(i) = 1.0;

  // Integration loop
  const auto rhs =
    [](const VectorType &src, VectorType &dst, const Number time) {
      const auto time_factor = Utilities::fixed_power<2>(std::sin(time));
      for (const auto i : src.locally_owned_elements())
        {
          dst(i) = src(i) * time_factor;
        }
    };
  for (auto it = 0; it < Niters; ++it)
    {
      const auto current_t = dt * it;
      integrator.perform_time_step(vct_y, current_t, dt, rhs);
    }

  // Print solution
  for (const auto i : vct_y.locally_owned_elements())
    deallog << i << " = " << vct_y(i) << std::endl;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  test<double, dealii::TrilinosWrappers::MPI::Vector>();
  deallog << std::endl;
  MPI_Barrier(comm);
  test<double,
       dealii::TrilinosWrappers::MPI::BlockVector,
       /*is_block_vector=*/true>();

  return 0;
}
