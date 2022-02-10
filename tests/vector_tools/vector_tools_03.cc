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


// Update ghost values with the help of LinearAlgebra::SharedMPI::Partitioner.

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include <deal.II/numerics/data_out.h>

#include <hyper.deal/base/mpi.h>
#include <hyper.deal/base/utilities.h>
#include <hyper.deal/grid/grid_generator.h>
#include <hyper.deal/lac/sm_vector.h>
#include <hyper.deal/numerics/vector_tools.h>

#include "../tests_functions.h"
#include "../tests_mf.h"

using namespace dealii;

template <int dim_x,
          int dim_v,
          int degree,
          int n_points,
          typename Number,
          typename VectorizedArrayType,
          typename VectorType>
void
test(const MPI_Comm &comm, const unsigned int n_refs)
{
  const auto sizes = hyperdeal::Utilities::decompose(
    dealii::Utilities::MPI::n_mpi_processes(comm));

  const unsigned int size_x = sizes.first;
  const unsigned int size_v = sizes.second;

  if (dealii::Utilities::pow<unsigned int>(2, n_refs) < size_x ||
      dealii::Utilities::pow<unsigned int>(2, n_refs) < size_v)
    {
      deallog << "Early return, since some processes have no cells!"
              << std::endl;
      return;
    }

  MPI_Comm comm_global =
    hyperdeal::mpi::create_rectangular_comm(comm, size_x, size_v);


  if (comm_global != MPI_COMM_NULL)
    {
      MPI_Comm comm_sm = MPI_COMM_SELF;

      hyperdeal::MatrixFreeWrapper<dim_x, dim_v, Number, VectorizedArrayType>
        matrixfree_wrapper(comm_global, comm_sm, size_x, size_v);

      hyperdeal::Parameters p;
      p.triangulation_type = "fullydistributed";
      p.degree             = degree;
      p.mapping_degree     = 1;
      p.do_collocation     = false;
      p.do_ghost_faces     = true;
      p.do_buffering       = false;
      p.use_ecl            = true;

      matrixfree_wrapper.init(p, [n_refs](auto &tria_x, auto &tria_v) {
        hyperdeal::GridGenerator::hyper_cube(
          tria_x, tria_v, true, n_refs, -1, +1);
      });

      const auto &matrix_free = matrixfree_wrapper.get_matrix_free();

      VectorType vec;
      matrix_free.initialize_dof_vector(vec, 0, false, true);

      VectorType vec_x;
      matrix_free.get_matrix_free_x().initialize_dof_vector(vec_x, 0);

      VectorType vec_v;
      matrix_free.get_matrix_free_v().initialize_dof_vector(vec_v, 0);

      static const int dim = dim_x + dim_v;

      const std::shared_ptr<Function<dim, Number>> solution =
        std::make_shared<DistanceFunction<dim, Number>>();

      hyperdeal::VectorTools::interpolate<degree, n_points>(
        solution, matrix_free, vec, 0, 0, 2, 2);

      hyperdeal::VectorTools::velocity_space_integration<degree, n_points>(
        matrix_free, vec_x, vec, 0, 0, 2);

      if (dealii::Utilities::MPI::this_mpi_process(
            matrixfree_wrapper.get_comm_column()) == 0)
        {
          vec_x.print(deallog.get_file_stream());

          DataOutBase::VtkFlags flags;
          flags.write_higher_order_cells = true;

          DataOut<dim_x> data_out;
          data_out.set_flags(flags);
          data_out.add_data_vector(
            matrix_free.get_matrix_free_x().get_dof_handler(),
            vec_x,
            "solution_x");
          data_out.build_patches(degree + 1);
          data_out.write_vtu_with_pvtu_record(
            "./", "result_x", 0, matrixfree_wrapper.get_comm_row());
        }

      hyperdeal::VectorTools::coordinate_space_integration<degree, n_points>(
        matrix_free, vec_v, vec, 0, 0, 2);

      if (dealii::Utilities::MPI::this_mpi_process(
            matrixfree_wrapper.get_comm_row()) == 0)
        {
          vec_v.print(deallog.get_file_stream());

          DataOutBase::VtkFlags flags;
          flags.write_higher_order_cells = true;

          DataOut<dim_v> data_out;
          data_out.set_flags(flags);
          data_out.add_data_vector(
            matrix_free.get_matrix_free_v().get_dof_handler(),
            vec_v,
            "solution_v");
          data_out.build_patches(degree + 1);
          data_out.write_vtu_with_pvtu_record(
            "./", "result_v", 0, matrixfree_wrapper.get_comm_column());
        }

      MPI_Comm_free(&comm_global);
    }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  for (unsigned int refs = 3; refs <= 3; refs++)
    test<2,
         2,
         3,
         4,
         double,
         dealii::VectorizedArray<double>,
         dealii::LinearAlgebra::SharedMPI::Vector<double>>(comm, refs);
}
