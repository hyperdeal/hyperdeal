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

#ifndef HYPERDEAL_PERFORMANCE_UTIL_DRIVER
#define HYPERDEAL_PERFORMANCE_UTIL_DRIVER

#include <hyper.deal/base/config.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/revision.h>

#include <hyper.deal/base/dynamic_convergence_table.h>
#include <hyper.deal/base/mpi.h>
#include <hyper.deal/base/utilities.h>

#include <fstream>

const MPI_Comm comm = MPI_COMM_WORLD;

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

#ifndef NUMBER_TYPE
#  define NUMBER_TYPE double
#endif

#ifndef MIN_DEGREE
#  define MIN_DEGREE 2
#endif

#ifndef MAX_DEGREE
#  define MAX_DEGREE 6
#endif

#ifndef MIN_DIM
#  define MIN_DIM 2
#endif

#ifndef MAX_DIM
#  define MAX_DIM 6
#endif

#ifndef MIN_SIMD_LENGTH
#  define MIN_SIMD_LENGTH 0
#endif

#ifndef MAX_SIMD_LENGTH
#  define MAX_SIMD_LENGTH 8
#endif

template <int dim_x,
          int dim_v,
          int degree,
          int n_points,
          typename Number,
          typename VectorizedArrayType>
void
test(const MPI_Comm &                    comm_global,
     const MPI_Comm &                    comm_sm,
     const unsigned int                  size_x,
     const unsigned int                  size_v,
     hyperdeal::DynamicConvergenceTable &table,
     const std::string                   file_name);

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
    prm.add_parameter("VLen", v_len);
    prm.add_parameter("Dim", dim);
    prm.add_parameter("Degree", degree);
    prm.add_parameter("PartitionX", partition_x);
    prm.add_parameter("PartitionV", partition_v);
    prm.add_parameter("UseVirtualTopology", use_virtual_topology);
    prm.add_parameter("UseSharedMemory", use_shared_memory);
    prm.add_parameter("Verbose", print_parameter);
    prm.leave_subsection();
  }

  unsigned int v_len                = 0;
  unsigned int dim                  = 4;
  unsigned int degree               = 3;
  unsigned int partition_x          = 1;
  unsigned int partition_v          = 1;
  bool         use_virtual_topology = true;
  bool         use_shared_memory    = true;
  bool         print_parameter      = false;
};

template <typename VectorizedArrayType,
          int dim_x,
          int dim_v,
          int degree,
          int n_points = degree + 1>
void
run_degree(const ParametersDriver &            param,
           hyperdeal::DynamicConvergenceTable &table,
           const std::string                   file_name)
{
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
      test<dim_x, dim_v, degree, n_points, NUMBER_TYPE, VectorizedArrayType>(
        comm_z,
        param.use_shared_memory ? comm_sm : MPI_COMM_SELF,
        size_x,
        size_v,
        table,
        file_name);

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

template <typename VectorizedArrayType, int dim_x, int dim_v>
void
run_dim(const ParametersDriver &            param,
        hyperdeal::DynamicConvergenceTable &table,
        const std::string                   file_name)
{
  // clang-format off
  switch(param.degree)
  {
#if MIN_DEGREE <= 2 && 2 <= MAX_DEGREE
    case 2: run_degree<VectorizedArrayType, dim_x, dim_v, 2>(param, table, file_name); break; 
#endif
#if MIN_DEGREE <= 3 && 3 <= MAX_DEGREE
    case 3: run_degree<VectorizedArrayType, dim_x, dim_v, 3>(param, table, file_name); break; 
#endif
#if MIN_DEGREE <= 4 && 4 <= MAX_DEGREE
    case 4: run_degree<VectorizedArrayType, dim_x, dim_v, 4>(param, table, file_name); break; 
#endif
#if MIN_DEGREE <= 5 && 5 <= MAX_DEGREE
    case 5: run_degree<VectorizedArrayType, dim_x, dim_v, 5>(param, table, file_name); break; 
#endif
#if MIN_DEGREE <= 6 && 6 <= MAX_DEGREE
    case 6: run_degree<VectorizedArrayType, dim_x, dim_v, 6>(param, table, file_name); break; 
#endif
    default:
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
  }
  // clang-format on
}

template <typename VectorizedArrayType>
void
run_simd(const ParametersDriver &            param,
         hyperdeal::DynamicConvergenceTable &table,
         const std::string                   file_name)
{
  // clang-format off
  switch(param.dim)
  {
#if MIN_DIM <= 2 && 2 <= MAX_DIM
    case 2: run_dim<VectorizedArrayType, 1, 1>(param, table, file_name); break; 
#endif
#if MIN_DIM <= 3 && 3 <= MAX_DIM
    case 3: run_dim<VectorizedArrayType, 2, 1>(param, table, file_name); break; 
#endif
#if MIN_DIM <= 4 && 4 <= MAX_DIM
    case 4: run_dim<VectorizedArrayType, 2, 2>(param, table, file_name); break; 
#endif
#if MIN_DIM <= 5 && 5 <= MAX_DIM
    case 5: run_dim<VectorizedArrayType, 3, 2>(param, table, file_name); break; 
#endif
#if MIN_DIM <= 6 && 6 <= MAX_DIM
    case 6: run_dim<VectorizedArrayType, 3, 3>(param, table, file_name); break; 
#endif
    default:
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
  }
  // clang-format on
}

void
run(hyperdeal::DynamicConvergenceTable &table, const std::string file_name)
{
  // read input parameter file
  const ParametersDriver param(
    file_name,
    dealii::ConditionalOStream(std::cout,
                               dealii::Utilities::MPI::this_mpi_process(comm) ==
                                 0));

  // clang-format off
  switch(param.v_len)
  {
    case 0: run_simd<dealii::VectorizedArray<NUMBER_TYPE>  >(param, table, file_name); break;
#if MIN_SIMD_LENGTH <= 1 && 1 <= MAX_SIMD_LENGTH
    case 1: run_simd<dealii::VectorizedArray<NUMBER_TYPE,1>>(param, table, file_name); break;
#endif
#if MIN_SIMD_LENGTH <= 2 && 2 <= MAX_SIMD_LENGTH
    case 2: run_simd<dealii::VectorizedArray<NUMBER_TYPE,2>>(param, table, file_name); break; 
#endif
#if MIN_SIMD_LENGTH <= 4 && 4 <= MAX_SIMD_LENGTH
    case 4: run_simd<dealii::VectorizedArray<NUMBER_TYPE,4>>(param, table, file_name); break; 
#endif
#if MIN_SIMD_LENGTH <= 8 && 8 <= MAX_SIMD_LENGTH
    case 8: run_simd<dealii::VectorizedArray<NUMBER_TYPE,8>>(param, table, file_name); break; 
#endif
    default:
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
  }
  // clang-format on
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

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_INIT;
      LIKWID_MARKER_THREADINIT;
#endif

      hyperdeal::DynamicConvergenceTable table;

      if (argc == 1)
        {
          if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
            printf("ERROR: No .json parameter files has been provided!\n");

          return 1;
        }

      for (int i = 1; i < argc; i++)
        {
          if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
            std::cout << std::string(argv[i]) << std::endl;

          run(table, std::string(argv[i]));
        }

      table.print(false);

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_CLOSE;
#endif
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

#endif