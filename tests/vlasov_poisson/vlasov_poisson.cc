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

#include <hyper.deal/base/config.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/revision.h>

#include <hyper.deal/base/dynamic_convergence_table.h>
#include <hyper.deal/base/mpi.h>
#include <hyper.deal/base/utilities.h>

#include "../../examples/vlasov_poisson/include/application.h"

const unsigned int dim_x    = 2;
const unsigned int dim_v    = 2;
const unsigned int degree   = 3;
const unsigned int n_points = degree + 1;
const MPI_Comm     comm     = MPI_COMM_WORLD;

using Number              = double;
using VectorizedArrayType = dealii::VectorizedArray<Number>;

using Problem = hyperdeal::vp::
  Application<dim_x, dim_v, degree, n_points, Number, VectorizedArrayType>;

struct ParametersDriver
{
  ParametersDriver()
  {}

  ParametersDriver(const std::string &               file_name,
                   const dealii::ConditionalOStream &pcout)
  {
    dealii::ParameterHandler prm;

    std::ifstream file;
    file.open(file_name);

    add_parameters(prm);

    prm.parse_input_from_json(file, true);

    if (print_parameter && pcout.is_active())
      prm.print_parameters(pcout.get_stream(),
                           dealii::ParameterHandler::OutputStyle::Text);

    file.close();
  }

  void
  add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("General");
    prm.add_parameter("DimX", dim_x);
    prm.add_parameter("DimV", dim_v);
    prm.add_parameter("DegreeX", degree_x);
    prm.add_parameter("DegreeV", degree_v);
    prm.add_parameter("PartitionX", partition_x);
    prm.add_parameter("PartitionV", partition_v);
    prm.add_parameter("UseVirtualTopology", use_virtual_topology);
    prm.add_parameter("Verbose", print_parameter);
    prm.leave_subsection();
  }

  unsigned int dim_x = 1;
  unsigned int dim_v = 1;

  unsigned int degree_x = 1;
  unsigned int degree_v = 1;

  unsigned int partition_x = 1;
  unsigned int partition_v = 1;

  bool use_virtual_topology = true;

  bool print_parameter = true;
};

void
run(hyperdeal::DynamicConvergenceTable &table, const std::string file_name)
{
  // read input parameter file
  const ParametersDriver param(
    file_name,
    dealii::ConditionalOStream(std::cout,
                               dealii::Utilities::MPI::this_mpi_process(comm) ==
                                 0));

  // check input parameters (TODO: at the moment, degree and dimension are
  // fixed at compile time)
  // clang-format off
  AssertThrow(dim_x == param.dim_x, dealii::ExcDimensionMismatch(dim_x, param.dim_x));
  AssertThrow(dim_v == param.dim_v, dealii::ExcDimensionMismatch(dim_v, param.dim_v));
  AssertThrow(degree == param.degree_x, dealii::ExcDimensionMismatch(degree, param.degree_x));
  AssertThrow(degree == param.degree_v, dealii::ExcDimensionMismatch(degree, param.degree_v));
  // clang-format on

  // partitions of x- and v-space
  const unsigned int size_x = param.partition_x;
  const unsigned int size_v = param.partition_v;

  // create rectangular communicator
  MPI_Comm comm_global =
    hyperdeal::mpi::create_rectangular_comm(comm, size_x, size_v);

  // only proceed if process part of new communicator
  if (comm_global != MPI_COMM_NULL)
    {
      // create communicator for shared memory (only create once since
      // expensive)
      MPI_Comm comm_sm = hyperdeal::mpi::create_sm(comm_global);

      // optionally create virtual topology (blocked Morton order)
      MPI_Comm comm_z =
        param.use_virtual_topology ?
          hyperdeal::mpi::create_z_order_comm(
            comm_global,
            {size_x, size_v},
            hyperdeal::Utilities::decompose(
              hyperdeal::mpi::n_procs_of_sm(comm_global, comm_sm))) :
          comm_global;

#ifdef DEBUG
      if (param.print_parameter)
        {
          hyperdeal::mpi::print_sm(comm_z, comm_sm);
          hyperdeal::mpi::print_new_order(comm, comm_z);
        }
#endif

      // process problem
      Problem problem(comm_z, comm_sm, size_x, size_v, table);
      problem.reinit(file_name);
      problem.solve();

      // free communicators
      if (param.use_virtual_topology)
        MPI_Comm_free(&comm_z);
      MPI_Comm_free(&comm_sm);
      MPI_Comm_free(&comm_global);

      // print results
      table.add_new_row();
      table.print(false);
    }
  else
    {
#ifdef DEBUG
      if (param.print_parameter)
        hyperdeal::mpi::print_new_order(comm, MPI_COMM_NULL);
#endif
    }
}

int
main(int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

      if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
        printf("deal.II git version %s on branch %s\n\n",
               DEAL_II_GIT_SHORTREV,
               DEAL_II_GIT_BRANCH);

      dealii::deallog.depth_console(0);

      hyperdeal::DynamicConvergenceTable table;
      run(table, SOURCE_DIR "/vlasov_poisson.json");
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
