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

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/data_out.h>

#include <hyper.deal/base/memory_consumption.h>
#include <hyper.deal/base/time_integrators.h>
#include <hyper.deal/base/time_loop.h>
#include <hyper.deal/base/timers.h>
#include <hyper.deal/lac/sm_vector.h>
#include <hyper.deal/matrix_free/matrix_free.h>
#include <hyper.deal/numerics/vector_tools.h>
#include <hyper.deal/operators/advection/advection_operation.h>
#include <hyper.deal/operators/advection/cfl.h>
#include <hyper.deal/operators/advection/velocity_field_view.h>

#include "derivative_container.h"
#include "diagnostics.h"
#include "parameters.h"
#include "poisson.h"
#include "velocity_field_view.h"

namespace hyperdeal
{
  namespace vp
  {
    template <int dim_x,
              int dim_v,
              int degree,
              int n_points,
              typename Number,
              typename VectorizedArrayType>
    class Application
    {
    public:
      static const int dim = dim_x + dim_v;

      using MF =
        hyperdeal::MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>;
      using VectorizedArrayTypeX = typename MF::VectorizedArrayTypeX;
      using VectorizedArrayTypeV = typename MF::VectorizedArrayTypeV;

      using VelocityFieldView =
        vp::PhaseSpaceVelocityFieldView<dim_x,
                                        dim_v,
                                        degree,
                                        n_points,
                                        Number,
                                        VectorizedArrayTypeX>;

      using PoissonMatrixType =
        LaplaceOperator<dim_x, degree, n_points, Number, VectorizedArrayType>;

      using VectorType  = dealii::LinearAlgebra::SharedMPI::Vector<Number>;
      using VectorType2 = dealii::LinearAlgebra::distributed::Vector<Number>;


      Application(const MPI_Comm           comm_global,
                  const MPI_Comm           comm_sm,
                  const unsigned int       size_x,
                  const unsigned int       size_v,
                  DynamicConvergenceTable &table)
        : comm_global(comm_global)
        , comm_sm(comm_sm)
        , comm_row(mpi::create_row_comm(comm_global, size_x, size_v))
        , comm_column(mpi::create_column_comm(comm_global, size_x, size_v))
        , matrix_free(comm_global, comm_sm, matrix_free_x, matrix_free_v)
        , advection_operation(matrix_free, table)
        , pcout(std::cout,
                dealii::Utilities::MPI::this_mpi_process(comm_global) == 0)
        , table(table)
      {}

      ~Application()
      {
        MPI_Comm_free(&comm_row);
        MPI_Comm_free(&comm_column);
      }

      void
      reinit(const std::string file_name)
      {
        // clang-format off
        pcout << "--------------------------------------------------------------------------------" << std::endl;
        pcout << "Setup:" << std::endl;
        // clang-format on

        table.set("info->procs",
                  dealii::Utilities::MPI::n_mpi_processes(comm_global));
        table.set("info->procs_x",
                  dealii::Utilities::MPI::n_mpi_processes(comm_row));
        table.set("info->procs_v",
                  dealii::Utilities::MPI::n_mpi_processes(comm_column));
        table.set("info->procs_sm",
                  dealii::Utilities::MPI::n_mpi_processes(comm_sm));


        const unsigned int degree_x   = degree;
        const unsigned int degree_v   = degree;
        const unsigned int n_points_x = n_points;
        const unsigned int n_points_v = n_points;

        table.set("info->dim_x", dim_x);
        table.set("info->dim_v", dim_v);
        table.set("info->degree_x", degree_x);
        table.set("info->degree_v", degree_v);


        const auto initializer =
          cases::get<dim_x, dim_v, degree, Number>(file_name, pcout);

        initializer->set_input_parameters(param, file_name, pcout);

        MemoryStatMonitor memory_stat_monitor(comm_global);

        // step 1: create two low-dimensional triangulations
        {
          pcout << "  - dealii::Triangulation (x/v)" << std::endl;
          memory_stat_monitor.monitor("triangulation");
          // clang-format off
          if(param.triangulation_type == "fullydistributed")
            {
              triangulation_x.reset(new dealii::parallel::fullydistributed::Triangulation<dim_x>(comm_row));
              triangulation_v.reset(new dealii::parallel::fullydistributed::Triangulation<dim_v>(comm_column));
            }
#ifdef DEAL_II_WITH_P4EST
          else if(param.triangulation_type == "distributed")
            {
              triangulation_x.reset(new dealii::parallel::distributed::Triangulation<dim_x>(comm_row,
            dealii::Triangulation<dim_x>::none,
            dealii::parallel::distributed::Triangulation<dim_x>::construct_multigrid_hierarchy));
              triangulation_v.reset(new dealii::parallel::distributed::Triangulation<dim_v>(comm_column,
            dealii::Triangulation<dim_v>::none,
            dealii::parallel::distributed::Triangulation<dim_v>::construct_multigrid_hierarchy));
            }
#endif
          else
            AssertThrow(false, dealii::ExcMessage("Unknown triangulation!"));
          // clang-format on

          initializer->create_grid(triangulation_x, triangulation_v);

          // output low-dimensional meshes (optional)
          if (param.output_grid)
            {
              dealii::GridOut grid_out;

              if (dealii::Utilities::MPI::this_mpi_process(comm_column) == 0)
                grid_out.write_mesh_per_processor_as_vtu(*this->triangulation_x,
                                                         "grid_x");

              if (dealii::Utilities::MPI::this_mpi_process(comm_row) == 0)
                grid_out.write_mesh_per_processor_as_vtu(*this->triangulation_v,
                                                         "grid_v");
            }
        }

        // step 2: create two low-dimensional dof-handler
        {
          pcout << "  - dealii::DoFHandler (x/v)" << std::endl;
          memory_stat_monitor.monitor("dofhandler");
          dof_handler_x.reset(new dealii::DoFHandler<dim_x>(*triangulation_x));
          dof_handler_v.reset(new dealii::DoFHandler<dim_v>(*triangulation_v));

          dealii::FE_DGQ<dim_x> fe_x(degree_x);
          dealii::FE_DGQ<dim_v> fe_v(degree_v);
          dof_handler_x->distribute_dofs(fe_x);
          dof_handler_x->distribute_mg_dofs();
          dof_handler_v->distribute_dofs(fe_v);

          table.set("info->size_x", dof_handler_x->n_dofs());
          table.set("info->size_v", dof_handler_v->n_dofs());
          table.set("info->size",
                    dof_handler_x->n_dofs() * dof_handler_v->n_dofs());
        }

        // step 3: setup two low-dimensional matrix-frees
        {
          memory_stat_monitor.monitor("dealii::matrixfree");
          dealii::AffineConstraints<Number> constraint;
          constraint.close();

          {
            pcout << "  - dealii::MatrixFree (x)" << std::endl;
            typename dealii::MatrixFree<dim_x, Number, VectorizedArrayTypeX>::
              AdditionalData additional_data;
            additional_data.mapping_update_flags =
              dealii::update_gradients | dealii::update_JxW_values |
              dealii::update_quadrature_points;
            additional_data.mapping_update_flags_inner_faces =
              dealii::update_gradients | dealii::update_JxW_values;
            additional_data.mapping_update_flags_boundary_faces =
              dealii::update_gradients | dealii::update_JxW_values |
              dealii::update_quadrature_points;
            additional_data.hold_all_faces_to_owned_cells = true;
            additional_data.mapping_update_flags_faces_by_cells =
              dealii::update_gradients | dealii::update_JxW_values |
              dealii::update_quadrature_points;
            dealii::MappingQGeneric<dim_x> mapping_x(param.mapping_degree_x);

            std::vector<const dealii::DoFHandler<dim_x> *> dof_handlers{
              &*dof_handler_x};
            std::vector<const dealii::AffineConstraints<Number> *> constraints{
              &constraint};

            std::vector<dealii::Quadrature<1>> quads;
            if (param.do_collocation)
              quads.push_back(dealii::QGaussLobatto<1>(n_points_x));
            else
              quads.push_back(dealii::QGauss<1>(n_points_x));
            quads.push_back(dealii::QGauss<1>(degree_x + 1));
            quads.push_back(dealii::QGaussLobatto<1>(degree_x + 1));

            // ensure that inner and boundary faces are not mixed in the
            // same macro cell
            if (param.use_ecl)
              dealii::MatrixFreeTools::categorize_by_boundary_ids(
                *triangulation_x, additional_data);

            matrix_free_x.reinit(
              mapping_x, dof_handlers, constraints, quads, additional_data);

            // Poisson solver: [TODO] not a great place here...
            {
              const bool use_multigrid = true; // [TODO]: parameter file

              poisson_matrix.initialize(
                matrix_free_x, initializer->get_poisson_problem_bc_type());

              if (!use_multigrid) // solve with PCG + Chebychev preconditioner
                {
                  poisson_solver.reset(
                    new PoissonSolver<PoissonMatrixType>(poisson_matrix));
                }
              else // solve with PCG + multigrid preconditioner
                {
                  const unsigned int min_level = 0;
                  const unsigned int max_level =
                    triangulation_x->n_global_levels() - 1;

                  auto level_matrix_free =
                    std::make_shared<dealii::MGLevelObject<
                      dealii::MatrixFree<dim_x, Number, VectorizedArrayTypeX>>>(
                      min_level, max_level);
                  auto mg_matrices =
                    std::make_shared<dealii::MGLevelObject<PoissonMatrixType>>(
                      min_level, max_level);

                  // initialize levels
                  for (unsigned int level = min_level; level <= max_level;
                       level++)
                    {
                      // ... initialize matrix_free
                      additional_data.mg_level = level;

                      if (param.use_ecl)
                        dealii::MatrixFreeTools::categorize_by_boundary_ids(
                          *triangulation_x, additional_data);

                      (*level_matrix_free)[level].reinit(mapping_x,
                                                         dof_handlers,
                                                         constraints,
                                                         quads,
                                                         additional_data);

                      // ... initialize level operator
                      (*mg_matrices)[level].initialize(
                        (*level_matrix_free)[level],
                        initializer->get_poisson_problem_bc_type());
                    }

                  poisson_solver.reset(new PoissonSolverMG<PoissonMatrixType>(
                    poisson_matrix, level_matrix_free, mg_matrices));
                }
            }
          }

          {
            pcout << "  - dealii::MatrixFree (v)" << std::endl;
            typename dealii::MatrixFree<dim_v, Number, VectorizedArrayTypeV>::
              AdditionalData additional_data;
            additional_data.mapping_update_flags =
              dealii::update_gradients | dealii::update_JxW_values |
              dealii::update_quadrature_points;
            additional_data.mapping_update_flags_inner_faces =
              dealii::update_gradients | dealii::update_JxW_values;
            additional_data.mapping_update_flags_boundary_faces =
              dealii::update_gradients | dealii::update_JxW_values |
              dealii::update_quadrature_points;
            additional_data.hold_all_faces_to_owned_cells = true;
            additional_data.mapping_update_flags_faces_by_cells =
              dealii::update_gradients | dealii::update_JxW_values |
              dealii::update_quadrature_points;
            dealii::MappingQGeneric<dim_v> mapping_v(param.mapping_degree_v);

            std::vector<const dealii::DoFHandler<dim_v> *> dof_handlers{
              &*dof_handler_v};
            std::vector<const dealii::AffineConstraints<Number> *> constraints{
              &constraint};
            std::vector<dealii::Quadrature<1>> quads;
            if (param.do_collocation)
              quads.push_back(dealii::QGaussLobatto<1>(n_points_v));
            else
              quads.push_back(dealii::QGauss<1>(n_points_v));

            quads.push_back(dealii::QGauss<1>(degree_v + 1));
            quads.push_back(dealii::QGaussLobatto<1>(degree_v + 1));

            // ensure that inner and boundary faces are not mixed in the
            // same macro cell
            if (param.use_ecl)
              dealii::MatrixFreeTools::categorize_by_boundary_ids(
                *triangulation_v, additional_data);

            matrix_free_v.reinit(
              mapping_v, dof_handlers, constraints, quads, additional_data);
          }
        }

        // step 4: setup tensor-product matrixfree
        {
          memory_stat_monitor.monitor("hyperdeal_matrixfree");
          pcout << "  - hyperdeal::MatrixFree" << std::endl;
          typename MF::AdditionalData ad;
          ad.do_ghost_faces    = param.do_ghost_faces;
          ad.do_buffering      = param.do_buffering;
          ad.use_ecl           = param.use_ecl;
          ad.overlapping_level = param.overlapping_level;

          matrix_free.reinit(ad);
        }

        // step 5: allocate memory for vectors (with only vct_Ti having ghost
        // value)
        {
          // clang-format off
          memory_stat_monitor.monitor("vector_solution");
          matrix_free.initialize_dof_vector(vct_solution, 0, false || !param.use_ecl, true);
          memory_stat_monitor.monitor("vector_Ki");
          matrix_free.initialize_dof_vector(vct_Ki, 0, false || !param.use_ecl, true);
          memory_stat_monitor.monitor("vector_Ti");
          matrix_free.initialize_dof_vector(vct_Ti, 0, true, true);
          // clang-format on

          // initialize vectors in x-space
          memory_stat_monitor.monitor("vector_particle_density");
          matrix_free_x.initialize_dof_vector(particle_density);
          particle_density = 0.0;
          particle_density.zero_out_ghosts();

          memory_stat_monitor.monitor("vector_potential");
          matrix_free_x.initialize_dof_vector(potential);
          potential = 0.0;
          potential.zero_out_ghosts();
        }

        // step 6: set initial condition in Gauss-Lobatto points (quad_no_x = 2,
        // quad_no_v = 2)
        {
          pcout << "  - advection::initial_condition" << std::endl;
          initializer->set_analytical_solution(analytical_solution);
          VectorTools::interpolate<degree, n_points>(
            analytical_solution, matrix_free, vct_solution, 0, 0, 2, 2);
        }

        // step 7: set boundary conditions
        {
          pcout << "  - advection::boundary_condition" << std::endl;
          boundary_descriptor.reset(
            new advection::BoundaryDescriptor<dim, Number>());
          initializer->set_boundary_conditions(boundary_descriptor);
        }

        // step 8: set (constant) velocity field
        {
          pcout << "  - advection::velocity_field_view" << std::endl;
          velocity_field =
            std::make_shared<VelocityFieldView>(matrix_free,
                                                negative_electric_field);
        }

        // step 9: initialize advection operator
        {
          pcout << "  - advection::operator" << std::endl;
          advection_operation.reinit(boundary_descriptor,
                                     velocity_field,
                                     param.advection_operation_parameters);
        }

        // step 10: time loop
        {
          pcout << "  - advection::time_loop" << std::endl;
          auto &time_loop_parameters = param.time_loop_parameters;

          // determine critical time step (CFL condition)
          const Number critical_time_step =
            hyperdeal::advection::compute_critical_time_step(
              *dof_handler_x.get(),
              *dof_handler_v.get(),
              initializer->get_transport_direction(),
              degree_x,
              n_points_x,
              comm_row,
              degree_v,
              n_points_v,
              comm_column);

          // determine minimal time step (parameter file vs. CFL)
          const Number dt = std::min(time_loop_parameters.time_step,
                                     param.cfl_number * critical_time_step /
                                       std::pow(degree, 1.5));

          // set time step s.t. final time step ends at final time
          time_loop_parameters.time_step =
            (time_loop_parameters.final_time -
             time_loop_parameters.start_time) /
            std::ceil((time_loop_parameters.final_time -
                       time_loop_parameters.start_time) /
                      dt);

          // setup time loop
          time_loop.reinit(time_loop_parameters);
        }

        {
          memory_stat_monitor.monitor("negative_electric_field");
          negative_electric_field.reinit(matrix_free_x);
        }

        // clang-format off
        pcout << "                                                                       ... done!" << std::endl;
        pcout << "--------------------------------------------------------------------------------" << std::endl << std::endl;
        // clang-format on

        {
          memory_stat_monitor.print(pcout);

          const auto &mem = memory_consumption();
          mem.print(comm_global, pcout);
        }
      }

      void
      solve()
      {
        LowStorageRungeKuttaIntegrator<Number, VectorType> time_integrator(
          vct_Ki, vct_Ti, param.rk_parameters.type, param.use_ecl);

        bool         clear_diag_file            = true;
        unsigned int time_step_counter          = 0;
        unsigned int n_total_poisson_iterations = 0;

#ifdef PERFORMANCE_TIMING
        bool performance_timing = true;
#else
        bool performance_timing = false;
#endif

        Timers timers(param.performance_log_all_calls);

        const unsigned int time_steps = time_loop.loop(
          vct_solution,
          [&](auto &      solution,
              const auto  cur_time,
              const auto  time_step,
              const auto &runnable) {
#ifdef PERFORMANCE_TIMING
            if (performance_timing)
              {
                if (time_step_counter == param.performance_warm_up_iterations)
                  {
                    if (time_step_counter != 0)
                      timers.reset();
                    n_total_poisson_iterations = 0;
                    MPI_Barrier(comm_global);
                    timers["id_total"].start();
                  }
                time_step_counter++;
              }
#endif

            ScopedTimerWrapper timer(timers, "id_stage");

            time_integrator.perform_time_step(solution,
                                              cur_time,
                                              time_step,
                                              runnable);
          },
          [&](const VectorType &src, VectorType &dst, const Number cur_time) {
            {
              // step 1: determine particle density ρ via integral over v-space
              ScopedTimerWrapper timer(timers, "id_step1");

              //...use Gauss-Lobatto points (quad_no_v = 2)
              VectorTools::velocity_space_integration<degree, n_points>(
                matrix_free,
                particle_density,
                src,
                0 /* dof_no_x*/,
                0 /*dof_no_v*/,
                2 /* quad_no_v */);
            }

            {
              // step 2: Take 1-particle_density (where we use the approximated
              //  one by the mean)
              ScopedTimerWrapper timer(timers, "id_step2");

              // subtract mean (why is this necessary?)
              if (poisson_matrix.is_singular())
                particle_density.add(-particle_density.mean_value());

              // test the particle density to obtain the right-hand-side
              // for the Poisson problem
              dealii::FEEvaluation<dim_x, degree, n_points, 1, Number> phi_x(
                matrix_free_x);

              matrix_free_x.template cell_loop<VectorType2, VectorType2>(
                [&](const auto &,
                    auto &      dst,
                    const auto &src,
                    const auto  range) mutable {
                  for (unsigned int cell = range.first; cell < range.second;
                       ++cell)
                    {
                      phi_x.reinit(cell);
                      phi_x.gather_evaluate(src, true, false);
                      for (unsigned int q = 0; q < phi_x.n_q_points; ++q)
                        phi_x.submit_value(-phi_x.get_value(q), q);
                      phi_x.integrate(true, false);
                      phi_x.set_dof_values(dst);
                    }
                },
                particle_density,
                particle_density);

              // subtract mean needed by Poisson solver
              if (poisson_matrix.is_singular())
                particle_density.add(-particle_density.mean_value());
            }

            {
              // step 3: solution of Poisson problem with inverted partial
              // density
              ScopedTimerWrapper timer(timers, "id_step3");

              n_total_poisson_iterations +=
                poisson_solver->solve(potential, particle_density);
            }

            {
              // step 4: derivative of potential gives the negative electric
              // field
              ScopedTimerWrapper timer(timers, "id_step4");

              negative_electric_field.update(matrix_free_x, potential);
            }

            {
              // step 5: advect in phase space
              ScopedTimerWrapper timer(timers, "id_step5");

              timers.enter("id_step5");
              {
                ScopedTimerWrapper timer(timers, "id_advection");

                if (performance_timing || param.performance_log_all_calls)
                  advection_operation.apply(dst, src, cur_time, &timers);
                else
                  advection_operation.apply(dst, src, cur_time);
              }
              timers.leave();
            }
          },
          [&](const Number cur_time) {
            if (!param.dignostics_enabled ||
                (cur_time != param.time_loop_parameters.start_time &&
                 static_cast<int>((cur_time + 0.00000000001 -
                                   param.time_loop_parameters.start_time) /
                                  param.dignostics_tick) ==
                   static_cast<int>((cur_time + 0.00000000001 -
                                     param.time_loop_parameters.start_time -
                                     param.time_loop_parameters.time_step) /
                                    param.dignostics_tick)))
              return; // nothing to do

            ScopedTimerWrapper timer(timers, "id_diagnostics");

            // compute electric energy [TODO] recompute potential!
            const auto en =
              compute_electric_energy<degree, n_points>(matrix_free_x,
                                                        potential);

            // run phase-space diagnostics
            const auto diag_val =
              phase_space_diagnostics<degree, n_points>(matrix_free,
                                                        vct_solution);

            if (pcout.is_active())
              fprintf(stdout, "   Time:%10.3e \n", cur_time);

            if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
              {
                std::ofstream diag_file;
                if (clear_diag_file == true)
                  {
                    diag_file.open(param.diag_file, std::ios::trunc);

                    diag_file
                      << "# time    energy components  mass  l2norm  kinetic energy  momentum "
                      << std::endl;

                    diag_file << std::setw(8) << std::fixed
                              << std::setprecision(3) << cur_time << "  "
                              << std::setprecision(16) << std::scientific << en
                              << " " << diag_val[0] << " " << diag_val[1] << " "
                              << diag_val[2] << " " << diag_val[3] << " "
                              << diag_val[4] << " " << diag_val[5] << std::endl;
                    clear_diag_file = false;
                  }
                else
                  {
                    diag_file.open(param.diag_file, std::ios::app);
                    diag_file << std::setw(8) << std::fixed
                              << std::setprecision(3) << cur_time << "  "
                              << std::setprecision(16) << std::scientific << en
                              << " " << diag_val[0] << " " << diag_val[1] << " "
                              << diag_val[2] << " " << diag_val[3] << " "
                              << diag_val[4] << " " << diag_val[5] << std::endl;
                  }
              }

            // temporal particle density vector since the actual particle
            // vector is overwritten in the Poisson solver
            VectorType2 particle_density_temp;
            matrix_free_x.initialize_dof_vector(particle_density_temp);

            // compute particle density and potiential again
            if (param.dignostics_vtk)
              {
                VectorTools::velocity_space_integration<degree, n_points>(
                  matrix_free,
                  particle_density,
                  vct_solution,
                  0 /* dof_no_x*/,
                  0 /*dof_no_v*/,
                  2 /* quad_no_v */);

                if (poisson_matrix.is_singular())
                  particle_density.add(-particle_density.mean_value());

                particle_density_temp.copy_locally_owned_data_from(
                  particle_density);

                dealii::FEEvaluation<dim_x, degree, n_points, 1, Number> phi_x(
                  matrix_free_x);

                matrix_free_x.template cell_loop<VectorType2, VectorType2>(
                  [&](const auto &,
                      auto &      dst,
                      const auto &src,
                      const auto  range) mutable {
                    for (unsigned int cell = range.first; cell < range.second;
                         ++cell)
                      {
                        phi_x.reinit(cell);
                        phi_x.gather_evaluate(src, true, false);
                        for (unsigned int q = 0; q < phi_x.n_q_points; ++q)
                          phi_x.submit_value(-phi_x.get_value(q), q);
                        phi_x.integrate(true, false);
                        phi_x.set_dof_values(dst);
                      }
                  },
                  particle_density,
                  particle_density);

                // subtract mean needed by Poisson solver
                if (poisson_matrix.is_singular())
                  particle_density.add(-particle_density.mean_value());

                potential = 0.0;
                poisson_solver->solve(potential, particle_density);
              }

            // output VTK files once
            if (dealii::Utilities::MPI::this_mpi_process(comm_column) == 0)
              {
                dealii::DataOutBase::VtkFlags flags;
                flags.write_higher_order_cells = true;

                dealii::DataOut<dim_x> data_out;
                data_out.set_flags(flags);

                data_out.attach_dof_handler(*dof_handler_x);
                data_out.add_data_vector(particle_density_temp,
                                         "particle_density");
                data_out.add_data_vector(potential, "potential");
                data_out.build_patches(
                  dealii::MappingQGeneric<dim_x>(3),
                  3,
                  dealii::DataOut<dim_x>::CurvedCellRegion::curved_inner_cells);
                const std::string filename =
                  "solution_" + std::to_string(time_step_counter) + ".vtu";
                data_out.write_vtu_in_parallel(filename, comm_row);
              }
          });

#ifdef PERFORMANCE_TIMING
        if (performance_timing)
          {
            MPI_Barrier(comm_global);
            timers["id_total"].stop();
            timers.print(comm_global, pcout);
          }
#endif


        {
#ifdef PERFORMANCE_TIMING
          if (performance_timing)
            {
              auto &timer = timers["id_stage"];
              AssertThrow(time_steps == time_step_counter,
                          dealii::StandardExceptions::ExcMessage(
                            "Mismatch in time step counter!"));
              AssertThrow((time_steps - param.performance_warm_up_iterations) ==
                            timer.get_counter(),
                          dealii::StandardExceptions::ExcMessage(
                            "Mismatch in time step counter!"));
            }
#endif

          table.set("info->time_steps",
                    time_steps - param.performance_warm_up_iterations);

#ifdef PERFORMANCE_TIMING
          if (performance_timing)
            {
              auto &timer = timers["id_stage"];
              table.set("throughput [MDoFs/s]",
                        dof_handler_x->n_dofs() * dof_handler_v->n_dofs() *
                          timer.get_counter() * time_integrator.n_stages() /
                          timer.get_accumulated_time());
              table.set("throughput [MDoFs/s/core]",
                        dof_handler_x->n_dofs() * dof_handler_v->n_dofs() *
                          timer.get_counter() * time_integrator.n_stages() /
                          timer.get_accumulated_time() /
                          dealii::Utilities::MPI::n_mpi_processes(comm_row) /
                          dealii::Utilities::MPI::n_mpi_processes(comm_column));
              table.set("avg. Poisson iterations",
                        static_cast<double>(n_total_poisson_iterations) /
                          timers["id_step3"].get_counter());
            }
#endif
        }
      }

      MemoryConsumption
      memory_consumption() const
      {
        MemoryConsumption mem("application");

        mem.insert("triangulation_x", triangulation_x->memory_consumption());
        mem.insert("triangulation_v", triangulation_v->memory_consumption());

        mem.insert("dof_handler_x", dof_handler_x->memory_consumption());
        mem.insert("dof_handler_v", dof_handler_v->memory_consumption());

        mem.insert("matrix_free_x", matrix_free_x.memory_consumption());
        mem.insert("matrix_free_v", matrix_free_v.memory_consumption());

        mem.insert(matrix_free.memory_consumption());

        MemoryConsumption mem_vec("vct");
        mem_vec.insert("solution", vct_solution.memory_consumption());
        mem_vec.insert("Ki", vct_Ki.memory_consumption());
        mem_vec.insert("Ti", vct_Ti.memory_consumption());

        mem_vec.insert("particle_density",
                       particle_density.memory_consumption());
        mem_vec.insert("potential", potential.memory_consumption());
        mem.insert(mem_vec);

        // [TODO]: poisson_matrix
        // [TODO]: poisson_solver

        mem.insert("negative_electric_field",
                   negative_electric_field.memory_consumption());

        return mem;
      }

    private:
      // communicators
      const MPI_Comm comm_global;
      const MPI_Comm comm_sm;
      MPI_Comm comm_row;    // should be const. but has to be freed at the end
      MPI_Comm comm_column; // the same here

      // x- and v-space objects
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>>
        triangulation_x;
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>>
                                                              triangulation_v;
      std::shared_ptr<dealii::DoFHandler<dim_x>>              dof_handler_x;
      std::shared_ptr<dealii::DoFHandler<dim_v>>              dof_handler_v;
      dealii::MatrixFree<dim_x, Number, VectorizedArrayTypeX> matrix_free_x;
      dealii::MatrixFree<dim_v, Number, VectorizedArrayTypeV> matrix_free_v;
      MF                                                      matrix_free;

      advection::AdvectionOperation<dim_x,
                                    dim_v,
                                    degree,
                                    n_points,
                                    Number,
                                    VectorType,
                                    VelocityFieldView,
                                    VectorizedArrayTypeX>
        advection_operation;


      dealii::ConditionalOStream pcout;
      DynamicConvergenceTable &  table;

      std::shared_ptr<dealii::Function<dim, Number>> analytical_solution;
      std::shared_ptr<hyperdeal::advection::BoundaryDescriptor<dim, Number>>
                                         boundary_descriptor;
      std::shared_ptr<VelocityFieldView> velocity_field;


      VectorType vct_solution, vct_Ki, vct_Ti;


      TimeLoop<Number, VectorType> time_loop;

      Parameters<Number> param;

      PoissonMatrixType poisson_matrix;


      // TODO: document
      VectorType2 particle_density, potential;

      std::shared_ptr<PoissonSolverBase<VectorType2>> poisson_solver;


      // negative electric field
      hyperdeal::vp::DerivativeContainer<dim_x,
                                         degree,
                                         n_points,
                                         Number,
                                         VectorizedArrayTypeX>
        negative_electric_field;
    };

  } // namespace vp
} // namespace hyperdeal
