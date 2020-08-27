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

// Test performance of components of advection operator run as a stand alone.


#define NUMBER_TYPE double
#define MIN_DEGREE 3
#define MAX_DEGREE 3
#define MIN_DIM 6
#define MAX_DIM 6
#define MIN_SIMD_LENGTH 0
#define MAX_SIMD_LENGTH 0

#include <hyper.deal/grid/grid_generator.h>
#include <hyper.deal/operators/advection/advection_operation.h>
#include <hyper.deal/operators/advection/cfl.h>
#include <hyper.deal/operators/advection/velocity_field_view.h>

#include "../tests/tests_mf.h"
#include "util/driver.h"

template <int dim_x, int dim_v>
struct Parameters
{
  Parameters(const std::string &               file_name,
             const dealii::ConditionalOStream &pcout)
    : n_subdivisions_x(dim_x, 0)
    , n_subdivisions_v(dim_v, 0)
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
    prm.add_parameter("Verbose", print_parameter);
    prm.leave_subsection();

    prm.enter_subsection("Performance");
    prm.add_parameter("Iterations", n_iterations);
    prm.add_parameter("IterationsWarmup", n_iterations_warmup);
    prm.leave_subsection();

    prm.enter_subsection("Case");

    prm.add_parameter("NRefinementsX", n_refinements_x);
    prm.add_parameter("NRefinementsV", n_refinements_v);

    prm.enter_subsection("NSubdivisionsX");
    if (dim_x >= 1)
      prm.add_parameter("X", n_subdivisions_x[0]);
    if (dim_x >= 2)
      prm.add_parameter("Y", n_subdivisions_x[1]);
    if (dim_x >= 3)
      prm.add_parameter("Z", n_subdivisions_x[2]);
    prm.leave_subsection();

    prm.enter_subsection("NSubdivisionsV");
    if (dim_v >= 1)
      prm.add_parameter("X", n_subdivisions_v[0]);
    if (dim_v >= 2)
      prm.add_parameter("Y", n_subdivisions_v[1]);
    if (dim_v >= 3)
      prm.add_parameter("Z", n_subdivisions_v[2]);
    prm.leave_subsection();

    prm.leave_subsection();
  }

  bool print_parameter = false;

  unsigned int n_iterations_warmup = 0;
  unsigned int n_iterations        = 10;

  unsigned int n_refinements_x = 0;
  unsigned int n_refinements_v = 0;

  std::vector<unsigned int> n_subdivisions_x;
  std::vector<unsigned int> n_subdivisions_v;
};

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
     const std::string                   file_name)
{
  (void)table;

  auto pcout = dealii::ConditionalOStream(
    std::cout, dealii::Utilities::MPI::this_mpi_process(comm_global) == 0);

  hyperdeal::MatrixFreeWrapper<dim_x, dim_v, Number, VectorizedArrayType>
    matrixfree_wrapper(comm_global, comm_sm, size_x, size_v);

  const Parameters<dim_x, dim_v> param(file_name, pcout);

  // clang-format off
  const dealii::Point< dim_x > px_1 = dim_x == 1 ? dealii::Point< dim_x >(0.0) : (dim_x == 2 ? dealii::Point< dim_x >(0.0, 0.0) : dealii::Point< dim_x >(0.0, 0.0, 0.0)); 
  const dealii::Point< dim_x > px_2 = dim_x == 1 ? dealii::Point< dim_x >(1.0) : (dim_x == 2 ? dealii::Point< dim_x >(1.0, 1.0) : dealii::Point< dim_x >(1.0, 1.0, 1.0)); 
  const dealii::Point< dim_v > pv_1 = dim_v == 1 ? dealii::Point< dim_v >(0.0) : (dim_v == 2 ? dealii::Point< dim_v >(0.0, 0.0) : dealii::Point< dim_v >(0.0, 0.0, 0.0)); 
  const dealii::Point< dim_v > pv_2 = dim_v == 1 ? dealii::Point< dim_v >(1.0) : (dim_v == 2 ? dealii::Point< dim_v >(1.0, 1.0) : dealii::Point< dim_v >(1.0, 1.0, 1.0));
  // clang-format on

  const bool do_periodic_x = true;
  const bool do_periodic_v = true;

  const hyperdeal::Parameters p(file_name, pcout);

  AssertThrow(p.degree == degree,
              dealii::StandardExceptions::ExcMessage(
                "Degrees " + std::to_string(p.degree) + " and " +
                std::to_string(degree) + " do not match!"));

  // clang-format off
  matrixfree_wrapper.init(p, [&](auto & tria_x, auto & tria_v){hyperdeal::GridGenerator::subdivided_hyper_rectangle(
    tria_x, tria_v,
    param.n_refinements_x, param.n_subdivisions_x, px_1, px_2, do_periodic_x, 
    param.n_refinements_v, param.n_subdivisions_v, pv_1, pv_2, do_periodic_v);});
  // clang-format on

  const auto &matrix_free = matrixfree_wrapper.get_matrix_free();

  using VectorType = dealii::LinearAlgebra::SharedMPI::Vector<Number>;

  using VelocityFieldView =
    hyperdeal::advection::ConstantVelocityFieldView<dim_x + dim_v,
                                                    Number,
                                                    VectorizedArrayType,
                                                    dim_x,
                                                    dim_v>;

  hyperdeal::advection::AdvectionOperation<dim_x,
                                           dim_v,
                                           degree,
                                           n_points,
                                           Number,
                                           VectorType,
                                           VelocityFieldView,
                                           VectorizedArrayType>
    advection_operation(matrix_free, table);

  auto boundary_descriptor = std::make_shared<
    hyperdeal::advection::BoundaryDescriptor<dim_x + dim_v, Number>>();

  auto velocity_field =
    std::make_shared<VelocityFieldView>(dealii::Tensor<1, dim_x + dim_v>());

  advection_operation.reinit(boundary_descriptor,
                             velocity_field,
                             false /*TODO*/);

  VectorType vec_src, vec_dst;
  matrix_free.initialize_dof_vector(vec_src, 0, true, true);
  matrix_free.initialize_dof_vector(vec_dst, 0, !p.use_ecl, true);

  hyperdeal::Timers timers(false);

  {
    timers.enter("apply");
    {
      // run for warm up
      for (unsigned int i = 0; i < param.n_iterations_warmup; i++)
        advection_operation.apply(vec_dst, vec_src, 0.0);
    }

    {
      timers.enter("withouttimers");
      // run without timers
      hyperdeal::ScopedTimerWrapper timer(timers, "total");
      for (unsigned int i = 0; i < param.n_iterations; i++)
        advection_operation.apply(vec_dst, vec_src, 0.0);
      timers.leave();
    }

    {
      hyperdeal::ScopedLikwidTimerWrapper likwid(
        std::string("apply") + "_" +
        std::to_string(dealii::Utilities::MPI::n_mpi_processes(comm_global)) +
        "_" + std::to_string(dealii::Utilities::MPI::n_mpi_processes(comm_sm)) +
        "_" + std::to_string(p.use_ecl) + "_" +
        std::to_string(VectorizedArrayType::size()) + "_" +
        std::to_string(degree) + "_" + std::to_string(dim_x + dim_v));

      timers.enter("withtimers");
      // run with timers
      hyperdeal::ScopedTimerWrapper timer(timers, "total");
      for (unsigned int i = 0; i < param.n_iterations; i++)
        advection_operation.apply(vec_dst, vec_src, 0.0, &timers);
      timers.leave();
    }
    timers.leave();
  }

  table.set("info->size [DoFs]", matrixfree_wrapper.n_dofs());
  table.set("info->ghost_size [DoFs]",
            Utilities::MPI::sum(vec_src.n_ghost_entries(), comm_global));

  table.set("info->dim_x", dim_x);
  table.set("info->dim_v", dim_v);
  table.set("info->degree", degree);

  table.set("info->procs",
            dealii::Utilities::MPI::n_mpi_processes(comm_global));
  table.set("info->procs_x",
            dealii::Utilities::MPI::n_mpi_processes(
              matrixfree_wrapper.get_comm_row()));
  table.set("info->procs_v",
            dealii::Utilities::MPI::n_mpi_processes(
              matrixfree_wrapper.get_comm_column()));
  table.set("info->procs_sm", dealii::Utilities::MPI::n_mpi_processes(comm_sm));

  table.set("throughput without timers [MDoFs/s]",
            matrixfree_wrapper.n_dofs() * param.n_iterations /
              timers["apply:withouttimers:total"].get_accumulated_time());

  table.set("throughput [MDoFs/s]",
            matrixfree_wrapper.n_dofs() * param.n_iterations /
              timers["apply:withtimers:total"].get_accumulated_time());

  table.set("apply:total [s]",
            timers["apply:withtimers:total"].get_accumulated_time() / 1e6 /
              param.n_iterations);

  std::vector<std::pair<std::string, std::string>> timer_labels;

  // clang-format off
  if(p.use_ecl)
  {
    timer_labels.emplace_back("apply:ECL:update_ghost_values", "apply:withtimers:ECL:update_ghost_values");
    timer_labels.emplace_back("apply:ECL:zero_out_ghosts"    , "apply:withtimers:ECL:zero_out_ghosts");
    timer_labels.emplace_back("apply:ECL:loop"               , "apply:withtimers:ECL:loop");
    timer_labels.emplace_back("apply:ECL:barrier"            , "apply:withtimers:ECL:barrier");
  }
  else
  {
    timer_labels.emplace_back("apply:FCL:advection"                     , "apply:withtimers:FCL:advection");
    timer_labels.emplace_back("apply:FCL:advection:update_ghost_values" , "apply:withtimers:FCL:advection:update_ghost_values");
    timer_labels.emplace_back("apply:FCL:advection:zero_out_ghosts"     , "apply:withtimers:FCL:advection:zero_out_ghosts");
    timer_labels.emplace_back("apply:FCL:advection:cell_loop"           , "apply:withtimers:FCL:advection:cell_loop");
    timer_labels.emplace_back("apply:FCL:advection:face_loop_x"         , "apply:withtimers:FCL:advection:face_loop_x");
    timer_labels.emplace_back("apply:FCL:advection:face_loop_v"         , "apply:withtimers:FCL:advection:face_loop_v");
    timer_labels.emplace_back("apply:FCL:advection:boundary_loop_x"     , "apply:withtimers:FCL:advection:boundary_loop_x");
    timer_labels.emplace_back("apply:FCL:advection:boundary_loop_v"     , "apply:withtimers:FCL:advection:boundary_loop_v");
    timer_labels.emplace_back("apply:FCL:advection:compress"            , "apply:withtimers:FCL:advection:compress");
    timer_labels.emplace_back("apply:FCL:mass"                          , "apply:withtimers:FCL:mass");
  }
  
  {
    std::vector<double> timing_values(timer_labels.size());
    
    for(unsigned int i = 0; i < timer_labels.size(); i++)
      timing_values[i] = timers[timer_labels[i].second].get_accumulated_time() / 1e6 / param.n_iterations;
   
    const auto timing_values_min_max_avg = Utilities::MPI::min_max_avg (timing_values, comm_global);
    
    for(unsigned int i = 0; i < timer_labels.size(); i++)
    {
        const auto min_max_avg = timing_values_min_max_avg[i];
        table.set(timer_labels[i].first + ":avg [s]", min_max_avg.avg);
        table.set(timer_labels[i].first + ":min [s]", min_max_avg.min);
        table.set(timer_labels[i].first + ":max [s]", min_max_avg.max);
    }
  }

  // clang-format on
}
