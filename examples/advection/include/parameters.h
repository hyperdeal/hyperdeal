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
#include <hyper.deal/operators/advection/advection_operation_parameters.h>

#include <fstream>

namespace hyperdeal
{
  namespace advection
  {
    template <typename Number>
    struct Parameters
    {
      void
      add_parameters(dealii::ParameterHandler &prm)
      {
        prm.enter_subsection("SpatialDiscretization");
        prm.add_parameter("MappingX",
                          mapping_degree_x,
                          "Degree of the mapping in x-space.");
        prm.add_parameter("MappingV",
                          mapping_degree_v,
                          "Degree of the mapping in v-space.");
        prm.add_parameter(
          "DoCollocation",
          do_collocation,
          "Perform numerical integration directly in Gauss-Lobatto points (collocation setup - no basis change needed).");
        prm.leave_subsection();

        prm.enter_subsection("Triangulation");
        prm.add_parameter("Type",
                          triangulation_type,
                          "Type of the triangulation.");
        prm.add_parameter(
          "OutputGrid",
          output_grid,
          "Output x- and v-triangulation to file so that it can be viewed in Paraview.");
        prm.leave_subsection();

        prm.enter_subsection("TemporalDiscretization");
        time_loop_parameters.add_parameters(prm);
        rk_parameters.add_parameters(prm);
        prm.add_parameter("CFLNumber",
                          cfl_number,
                          "The CFL number limiting the time steps.");
        prm.add_parameter("DiagnosticsEnabled",
                          dignostics_enabled,
                          "Perform diagnostics (post-processing).");
        prm.add_parameter("DiagnosticsTick",
                          dignostics_tick,
                          "Perform diagnostics after this interveral.");
        prm.add_parameter(
          "PerformanceLogAllCalls",
          performance_log_all_calls,
          "Save all measured timings (extra memory consumption) or only store min/max/avg.");
        prm.add_parameter(
          "PerformanceLogAllCallsPrefix",
          performance_log_all_calls_prefix,
          "Save all measured timings (extra memory consumption) or only store min/max/avg.");
        prm.add_parameter(
          "DiagnosticsWarmUpIterations",
          performance_warm_up_iterations,
          "Number of warm-up iterations (these iterations are ignored for the timings).");
        prm.leave_subsection();

        prm.enter_subsection("AdvectionOperation");
        advection_operation_parameters.add_parameters(prm);
        prm.leave_subsection();

        prm.enter_subsection("Matrixfree");
        prm.add_parameter(
          "GhostFaces",
          do_ghost_faces,
          "Use ghost faces (advection) or ghost cells (Laplace).");
        prm.add_parameter("DoBuffering",
                          do_buffering,
                          "Use buffering-mode of shared memory vector.");
        prm.add_parameter(
          "UseECL",
          use_ecl,
          "Use element-centric loops (ECL) or face-centric loops (FCL).");
        prm.add_parameter(
          "OverlappingLevel",
          overlapping_level,
          "Level of overlapping of communication and computation.");
        prm.leave_subsection();

        prm.enter_subsection("General");
        prm.add_parameter("Verbose",
                          print_parameter,
                          "Print the parameter after parsing.");
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

      // ... advection operation
      AdvectionOperationParamters advection_operation_parameters;

      // ... diagnostic
      bool   dignostics_enabled = true;
      Number dignostics_tick    = 0.1;

      // ... performance
      bool         performance_log_all_calls        = false;
      std::string  performance_log_all_calls_prefix = "test";
      unsigned int performance_warm_up_iterations   = 0;

      // matrix-free
      bool         do_ghost_faces    = true;
      bool         do_buffering      = false;
      bool         use_ecl           = true;
      unsigned int overlapping_level = 0;

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
  } // namespace advection
} // namespace hyperdeal

#include "../cases/hyperrectangle.h"
#include "../cases/torus_hyperball.h"

namespace hyperdeal
{
  namespace advection
  {
    namespace cases
    {
      struct Parameters
      {
        Parameters(std::string case_name)
          : case_name(case_name)
        {}

        Parameters(const std::string &               file_name,
                   const dealii::ConditionalOStream &pcout)
          : case_name("dummy")
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
          prm.add_parameter(
            "Case",
            case_name,
            "Name of the case to be run (the section Case contains the parameters of this case).");
          prm.add_parameter("Verbose",
                            print_parameter,
                            "Print the parameter after parsing.");
          prm.leave_subsection();
        }

        std::string case_name;
        bool        print_parameter = true;
      };

      template <int dim_x, int dim_v, int degree, typename Number>
      std::shared_ptr<
        hyperdeal::advection::Initializer<dim_x, dim_v, degree, Number>>
      get(const std::string &case_name)
      {
        // clang-format off
        std::shared_ptr<hyperdeal::advection::Initializer<dim_x, dim_v, degree, Number>> initializer;
        if(case_name == "hyperrectangle")
          initializer.reset(new hyperdeal::advection::hyperrectangle::Initializer<dim_x, dim_v, degree, Number>);
        else if(case_name == "torus")
          initializer.reset(new hyperdeal::advection::torus_hyperball::Initializer<dim_x, dim_v, degree, Number>);
        else
          AssertThrow(false, dealii::ExcMessage("This case does not exist!"));
        // clang-format on

        return initializer;
      }

      template <int dim_x, int dim_v, int degree, typename Number>
      std::shared_ptr<
        hyperdeal::advection::Initializer<dim_x, dim_v, degree, Number>>
      get(const std::string &file_name, const dealii::ConditionalOStream &pcout)
      {
        Parameters params(file_name, pcout);

        return get<dim_x, dim_v, degree, Number>(params.case_name);
      }

    } // namespace cases
  }   // namespace advection
} // namespace hyperdeal

#endif
