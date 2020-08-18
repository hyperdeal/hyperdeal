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

#ifndef HYPERDEAL_ADVECTION_CASES_HYPERRECTANGLE
#define HYPERDEAL_ADVECTION_CASES_HYPERRECTANGLE

#include <hyper.deal/grid/grid_generator.h>

#include "../include/parameters.h"

namespace hyperdeal
{
  namespace advection
  {
    namespace hyperrectangle
    {
      template <int DIM, typename Number = double>
      class ExactSolution : public dealii::Function<DIM, Number>
      {
      public:
        ExactSolution(const double time = 0.)
          : dealii::Function<DIM, Number>(1, time)
          , wave_number(2.)
        {
          advection[0] = 1.;
          if (DIM > 1)
            advection[1] = 0.15;
          if (DIM > 2)
            advection[2] = -0.05;
        }

        virtual double
        value(const dealii::Point<DIM> &p, const unsigned int = 1) const
        {
          double                       t        = this->get_time();
          const dealii::Tensor<1, DIM> position = p - t * advection;
          double                       result =
            std::sin(wave_number * position[0] * dealii::numbers::PI);
          for (unsigned int d = 1; d < DIM; ++d)
            result *= std::cos(wave_number * position[d] * dealii::numbers::PI);
          return result;
        }

        dealii::Tensor<1, DIM>
        get_transport_direction() const
        {
          return advection;
        }

      private:
        dealii::Tensor<1, DIM> advection;
        const double           wave_number;
      };

      template <int dim_x, int dim_v, int degree, typename Number>
      class Initializer
        : public hyperdeal::advection::Initializer<dim_x, dim_v, degree, Number>
      {
      public:
        Initializer()
          : n_subdivisions_x(dim_x, 4)
          , n_subdivisions_v(dim_v, 4)
        {}

        void
        add_parameters(dealii::ParameterHandler &prm)
        {
          prm.enter_subsection("Case");

          prm.add_parameter("NRefinementsX",
                            n_refinements_x,
                            "x-space: number of global refinements.");
          prm.add_parameter("NRefinementsV",
                            n_refinements_v,
                            "v-space: number of global refinements.");

          prm.add_parameter("PeriodicX",
                            do_periodic_x,
                            "x-space: apply periodicity.");
          prm.add_parameter("PeriodicV",
                            do_periodic_v,
                            "v-space: apply periodicity.");

          prm.enter_subsection("NSubdivisionsX");
          if (dim_x >= 1)
            prm.add_parameter(
              "X",
              n_subdivisions_x[0],
              "x-space: number of subdivisions in x-direction.");
          if (dim_x >= 2)
            prm.add_parameter(
              "Y",
              n_subdivisions_x[1],
              "x-space: number of subdivisions in y-direction.");
          if (dim_x >= 3)
            prm.add_parameter(
              "Z",
              n_subdivisions_x[2],
              "x-space: number of subdivisions in z-direction.");
          prm.leave_subsection();

          prm.enter_subsection("NSubdivisionsV");
          if (dim_v >= 1)
            prm.add_parameter(
              "X",
              n_subdivisions_v[0],
              "v-space: number of subdivisions in x-direction.");
          if (dim_v >= 2)
            prm.add_parameter(
              "Y",
              n_subdivisions_v[1],
              "v-space: number of subdivisions in y-direction.");
          if (dim_v >= 3)
            prm.add_parameter(
              "Z",
              n_subdivisions_v[2],
              "v-space: number of subdivisions in z-direction.");
          prm.leave_subsection();

          prm.add_parameter("OrientationX",
                            orientation_x,
                            "x-space: number of global refinements.");
          prm.add_parameter("OrientationV",
                            orientation_v,
                            "v-space: number of global refinements.");

          prm.leave_subsection();
        }

        void
        create_grid(
          std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>> &tria_x,
          std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>> &tria_v)
        {
          dealii::Point<dim_x> p1_x;
          for (unsigned int i = 0; i < dim_x; i++)
            p1_x[i] = left;
          dealii::Point<dim_x> p2_x;
          for (unsigned int i = 0; i < dim_x; i++)
            p2_x[i] = right;
          dealii::Point<dim_v> p1_v;
          for (unsigned int i = 0; i < dim_v; i++)
            p1_v[i] = left;
          dealii::Point<dim_v> p2_v;
          for (unsigned int i = 0; i < dim_v; i++)
            p2_v[i] = right;

            // clang-format off
#if true
          hyperdeal::GridGenerator::subdivided_hyper_rectangle(tria_x, tria_v, 
              n_refinements_x, n_subdivisions_x, p1_x, p2_x, do_periodic_x,
              n_refinements_v, n_subdivisions_v, p1_v, p2_v, do_periodic_v);
#elif false
          hyperdeal::GridGenerator::orientated_hyper_cube(tria_x, tria_v, 
              n_refinements_x, p1_x, p2_x, do_periodic_x, orientation_x,
              n_refinements_v, p1_v, p2_v, do_periodic_v, orientation_v);
#else
          hyperdeal::GridGenerator::subdivided_hyper_ball(tria_x, tria_v, 
              n_refinements_x, p1_x, p2_x, do_periodic_x,
              n_refinements_v, p1_v, p2_v, do_periodic_v);
#endif
          // clang-format on
        }

        void
        set_boundary_conditions(
          std::shared_ptr<BoundaryDescriptor<dim_x + dim_v, Number>>
            boundary_descriptor)
        {
          boundary_descriptor->dirichlet_bc[0] =
            std::shared_ptr<dealii::Function<dim_x + dim_v, Number>>(
              new ExactSolution<dim_x + dim_v, Number>());

          // TODO: hack for 1D
          boundary_descriptor->dirichlet_bc[1] =
            std::shared_ptr<dealii::Function<dim_x + dim_v, Number>>(
              new ExactSolution<dim_x + dim_v, Number>());
        }

        void
        set_analytical_solution(
          std::shared_ptr<dealii::Function<dim_x + dim_v, Number>>
            &analytical_solution)
        {
          analytical_solution.reset(new ExactSolution<dim_x + dim_v, Number>());
        }

        dealii::Tensor<1, dim_x + dim_v>
        get_transport_direction()
        {
          return ExactSolution<dim_x + dim_v, Number>()
            .get_transport_direction();
        }

      private:
        const double left  = -1.0;
        const double right = +1.0;

        unsigned int n_refinements_x = 0;
        unsigned int n_refinements_v = 0;
        bool         do_periodic_x   = true;
        bool         do_periodic_v   = true;

        std::vector<unsigned int> n_subdivisions_x;
        std::vector<unsigned int> n_subdivisions_v;

        unsigned int orientation_x = 0;
        unsigned int orientation_v = 0;
      };
    } // namespace hyperrectangle
  }   // namespace advection
} // namespace hyperdeal

#endif
