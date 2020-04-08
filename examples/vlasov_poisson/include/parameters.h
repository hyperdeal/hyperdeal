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

#ifndef HYPERDEAL_ADVECTION_PARAMETERS
#define HYPERDEAL_ADVECTION_PARAMETERS

#include <hyper.deal/base/time_integrators_parameters.h>
#include <hyper.deal/base/time_loop_parameters.h>

#include <fstream>

namespace hyperdeal
{
  namespace vp
  {
    template <typename Number>
    struct Parameters
    {
      void
      add_parameters(dealii::ParameterHandler &prm)
      {
        prm.enter_subsection("SpatialDiscretization");
        prm.add_parameter("MappingX", mapping_degree_x);
        prm.add_parameter("MappingV", mapping_degree_v);
        prm.add_parameter("DoCollocation", do_collocation);
        prm.leave_subsection();

        prm.enter_subsection("Triangulation");
        prm.add_parameter("Type", triangulation_type);
        prm.add_parameter("OutputGrid", output_grid);
        prm.leave_subsection();

        prm.enter_subsection("TemporalDiscretization");
        time_loop_parameters.add_parameters(prm);
        rk_parameters.add_parameters(prm);
        prm.add_parameter("CFLNumber", cfl_number);
        prm.add_parameter("DiagnosticsEnabled", dignostics_enabled);
        prm.add_parameter("DiagnosticsTick", dignostics_tick);
        prm.add_parameter("DiagnosticsFileName", diag_file);
        prm.add_parameter("PerformanceLogAllCalls", performance_log_all_calls);
        prm.add_parameter("DiagnosticsWarmUpIterations",
                          performance_warm_up_iterations);
        prm.leave_subsection();

        prm.enter_subsection("Matrixfree");
        prm.add_parameter("GhostFaces", do_ghost_faces);
        prm.add_parameter("DoBuffering", do_buffering);
        prm.add_parameter("UseECL", use_ecl);
        prm.leave_subsection();

        prm.enter_subsection("General");
        prm.add_parameter("Verbose", print_parameter);
        prm.leave_subsection();
      }

      // discretization
      unsigned int mapping_degree_x = 1;
      unsigned int mapping_degree_v = 1;
      bool         do_collocation   = false;

      // triangulation
      std::string triangulation_type = "fullydistributed";
      bool        output_grid        = false;

      // time-loop
      TimeLoopParamters<Number>               time_loop_parameters;
      LowStorageRungeKuttaIntegratorParamters rk_parameters;

      // ... CFL-condition
      Number cfl_number = 0.3;

      // ... diagnostic
      bool   dignostics_enabled = true;
      Number dignostics_tick    = 0.1;
      std::string diag_file = "time_history_diagnostics.out";

      // ... performance
      bool         performance_log_all_calls      = false;
      unsigned int performance_warm_up_iterations = 0;

      // matrix-free
      bool do_ghost_faces = true;
      bool do_buffering   = false;
      bool use_ecl        = true;

      bool print_parameter = true;
    };


    template <int dim_x, int dim_v, int degree, typename Number>
    class Initializer
    {
    public:
      void
      set_input_parameters(Parameters<Number> &              param,
                           const std::string                 file_name,
                           const dealii::ConditionalOStream &pcout)
      {
        dealii::ParameterHandler prm;

        std::ifstream file;
        file.open(file_name);

        // add general parameters
        param.add_parameters(prm);

        // add case-specific parameters
        this->add_parameters(prm);

        prm.parse_input_from_json(file, true);

        if (param.print_parameter && pcout.is_active())
          prm.print_parameters(pcout.get_stream(),
                               dealii::ParameterHandler::OutputStyle::Text);

        file.close();
      }

      virtual void
      add_parameters(dealii::ParameterHandler &prm)
      {
        AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
        (void)prm;
      }

      virtual void
      create_grid(std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>>
                    &triangulation_x,
                  std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>>
                    &triangulation_v)
      {
        AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
        (void)triangulation_x;
        (void)triangulation_v;
      }


      virtual void
      set_boundary_conditions(
        std::shared_ptr<
          hyperdeal::advection::BoundaryDescriptor<dim_x + dim_v, Number>>
          boundary_descriptor)
      {
        AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
        (void)boundary_descriptor;
      }

      virtual void
      set_analytical_solution(
        std::shared_ptr<dealii::Function<dim_x + dim_v, Number>>
          &analytical_solution)
      {
        AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
        (void)analytical_solution;
      }

      virtual dealii::Tensor<1, dim_x + dim_v>
      get_transport_direction()
      {
        AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
        return dealii::Tensor<1, dim_x + dim_v>();
      }
    };
  } // namespace vp
} // namespace hyperdeal

#include "../cases/hyperrectangle.h"

namespace hyperdeal
{
  namespace vp
  {
    namespace cases
    {
      template <int dim_x, int dim_v, int degree, typename Number>
      std::shared_ptr<hyperdeal::vp::Initializer<dim_x, dim_v, degree, Number>>
      get(const std::string &file_name, const dealii::ConditionalOStream &pcout)
      {
        std::string case_name       = "dummy";
        bool        print_parameter = true;

        dealii::ParameterHandler prm;

        std::ifstream file;
        file.open(file_name);

        prm.enter_subsection("General");
        prm.add_parameter("Case", case_name);
        prm.add_parameter("Verbose", print_parameter);
        prm.leave_subsection();

        prm.parse_input_from_json(file, true);

        if (print_parameter && pcout.is_active())
          prm.print_parameters(pcout.get_stream(),
                               dealii::ParameterHandler::OutputStyle::Text);

        file.close();

        // clang-format off
        std::shared_ptr<hyperdeal::vp::Initializer<dim_x, dim_v, degree, Number>> initializer;
        if(case_name == "hyperrectangle")
          initializer.reset(new hyperdeal::vp::hyperrectangle::Initializer<dim_x, dim_v, degree, Number>);
        else
          AssertThrow(false, dealii::ExcMessage("This case does not exist!"));
        // clang-format on

        return initializer;
      }

    } // namespace cases
  }   // namespace vp
} // namespace hyperdeal

#endif
