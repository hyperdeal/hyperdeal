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


// Test hyperdeal::Utilities::decompose.

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include "../tests.h"

using namespace dealii;

#include <hyper.deal/../../examples/vlasov_poisson/include/poisson.h>

const MPI_Comm comm = MPI_COMM_WORLD;

template <int dim,
          int fe_degree,
          int n_q_points_1d            = fe_degree + 1,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
void
test(const bool do_test_multigrid)
{
  using MatrixFreeType = MatrixFree<dim, Number, VectorizedArrayType>;
  using VectorType     = LinearAlgebra::distributed::Vector<Number>;
  using MatrixType =
    LaplaceOperator<dim, fe_degree, n_q_points_1d, Number, VectorizedArrayType>;

  // create triangulation
  std::shared_ptr<Triangulation<dim>> tria;

  auto create_grid = [](auto &tria) {
    dealii::GridGenerator::torus(tria, 6.2, 2.0);
    ;
    tria.refine_global(2);
  };

  if (true)
    {
      dealii::Triangulation<dim> tria_serial(
        dealii::Triangulation<dim>::limit_level_difference_at_vertices);
      create_grid(tria_serial);

      dealii::GridTools::partition_triangulation_zorder(
        dealii::Utilities::MPI::n_mpi_processes(comm), tria_serial, false);
      dealii::GridTools::partition_multigrid_levels(tria_serial);

      const auto construction_data = dealii::TriangulationDescription::
        Utilities::create_description_from_triangulation(
          tria_serial,
          comm,
          dealii::TriangulationDescription::Settings::
            construct_multigrid_hierarchy);

      tria.reset(new parallel::fullydistributed::Triangulation<dim>(comm));

      const auto manifold_ids = tria_serial.get_manifold_ids();
      for (const auto manifold_id : manifold_ids)
        if (manifold_id != dealii::numbers::flat_manifold_id)
          {
            auto manifold = tria_serial.get_manifold(manifold_id).clone();

            if (auto temp =
                  dynamic_cast<dealii::TransfiniteInterpolationManifold<dim> *>(
                    manifold.get()))
              temp->initialize(*tria);

            tria->set_manifold(manifold_id, *manifold);
          }

      tria->create_triangulation(construction_data);
    }
  else
    {
      tria.reset(new dealii::Triangulation<dim>(
        dealii::Triangulation<
          dim>::MeshSmoothing::limit_level_difference_at_vertices));
      create_grid(*tria);
    }

  // create dof handler
  DoFHandler<dim> dof(*tria);
  FE_DGQ<dim>     fe(fe_degree);
  dof.distribute_dofs(fe);
  dof.distribute_mg_dofs();

  // create matrix free
  MatrixFreeType matrix_free;

  QGauss<1>     quad(n_q_points_1d);
  MappingQ<dim> mapping(fe_degree + 1);

  AffineConstraints<Number> constraints;
  constraints.close();

  typename MatrixFreeType::AdditionalData data;
  data.mapping_update_flags_inner_faces = update_gradients | update_JxW_values;
  data.mapping_update_flags_boundary_faces =
    update_gradients | update_JxW_values;

  matrix_free.reinit(mapping, dof, constraints, quad, data);

  // create vectors
  LinearAlgebra::distributed::Vector<Number> sol, in;
  matrix_free.initialize_dof_vector(sol);
  matrix_free.initialize_dof_vector(in);

  in = 1.0;

  // create operator
  MatrixType fine_matrix;
  fine_matrix.initialize(matrix_free);

  std::shared_ptr<PoissonSolverBase<VectorType>> solver;

  // create solver and solve
  if (do_test_multigrid == false) // solve with PCG + Chebychev preconditioner
    {
      solver.reset(new PoissonSolver<MatrixType>(fine_matrix));
    }
  else // solve with PCG + multigrid preconditioner
    {
      const unsigned int min_level = 0;
      const unsigned int max_level = tria->n_global_levels() - 1;

      auto level_matrix_free_ =
        std::make_shared<MGLevelObject<MatrixFreeType>>(min_level, max_level);
      auto mg_matrices_ =
        std::make_shared<MGLevelObject<MatrixType>>(min_level, max_level);

      auto &level_matrix_free = *level_matrix_free_;
      auto &mg_matrices       = *mg_matrices_;

      // initialize levels
      for (unsigned int level = min_level; level <= max_level; level++)
        {
          // ... initialize matrix_free
          data.mg_level = level;
          level_matrix_free[level].reinit(
            mapping, dof, constraints, quad, data);

          // ... initialize level operator
          mg_matrices[level].initialize(level_matrix_free[level]);
        }

      solver.reset(new PoissonSolverMG<MatrixType>(fine_matrix,
                                                   level_matrix_free_,
                                                   mg_matrices_));
    }

  solver->solve(sol, in);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  for (unsigned int i = 0; i <= 1; i++)
    {
      test<3, 1, 2, double>(true);
      test<3, 2, 3, double>(true);
    }
}
