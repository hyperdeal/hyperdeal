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

#ifndef HYPERDEAL_ADVECTION_CASES_TORUS_HYPERBALL
#define HYPERDEAL_ADVECTION_CASES_TORUS_HYPERBALL

#include <hyper.deal/grid/grid_generator.h>

#include "../include/parameters.h"

namespace hyperdeal
{
  namespace advection
  {
    namespace torus_hyperball
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
          if (DIM > 3)
            advection[3] = -0.10;
          if (DIM > 4)
            advection[4] = -0.15;
          if (DIM > 5)
            advection[5] = 0.5;
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

          prm.leave_subsection();
        }

        void
        create_grid(
          std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>> &tria_x,
          std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>> &tria_v)
        {
          const auto fu_x = [&](auto &tria) {
            if constexpr(dim_x == 3)
              dealii::GridGenerator::torus(tria, 2.0, 0.5);
            tria.refine_global(n_refinements_x);
          };
          const auto fu_v = [&](auto &tria) {
            if constexpr(dim_v == 3)
              dealii::GridGenerator::hyper_ball_balanced(tria);
            tria.refine_global(n_refinements_v);
          };

          hyperdeal::GridGenerator::construct_tensor_product<dim_x, dim_v>(
            tria_x, tria_v, fu_x, fu_v);
        }

        void
        set_boundary_conditions(
          std::shared_ptr<BoundaryDescriptor<dim_x + dim_v, Number>>
            boundary_descriptor)
        {
#if false
          boundary_descriptor->dirichlet_bc[0] =
            std::shared_ptr<dealii::Function<dim_x + dim_v, Number>>(
              new ExactSolution<dim_x + dim_v, Number>());

          // TODO: hack for 1D
          boundary_descriptor->dirichlet_bc[1] =
            std::shared_ptr<dealii::Function<dim_x + dim_v, Number>>(
              new ExactSolution<dim_x + dim_v, Number>());
#else
          boundary_descriptor->homogeneous_dirichlet_bc.insert(0);
          boundary_descriptor->homogeneous_dirichlet_bc.insert(1);
#endif
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
        unsigned int n_refinements_x = 0;
        unsigned int n_refinements_v = 0;
      };
    } // namespace torus_hyperball
  }   // namespace advection
} // namespace hyperdeal

#endif
