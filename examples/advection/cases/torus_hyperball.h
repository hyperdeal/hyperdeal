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

#include <deal.II/grid/manifold_lib.h>

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
          std::array<Number, 6> temp = {{1., 0.15, -0.05, -0.10, -0.15, 0.5}};

          for (int i = 0; i < DIM; ++i)
            advection[i] = temp[i];
        }

        virtual double
        value(const dealii::Point<DIM> &p,
              const unsigned int = 1) const override
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
        void
        add_parameters(dealii::ParameterHandler &prm) override
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
          override
        {
          const auto fu_x = [&](auto &tria) {
            if constexpr (dim_x == 3)
              {
                const double torus_R = 6.2; // ITER
                const double torus_r = 2.0;
                dealii::GridGenerator::torus(tria, torus_R, torus_r);

                tria.reset_all_manifolds();
                tria.set_manifold(1,
                                  dealii::TorusManifold<3>(torus_R, torus_r));
                tria.set_manifold(0,
                                  dealii::CylindricalManifold<3>(
                                    dealii::Tensor<1, 3>({0., 1., 0.}),
                                    dealii::Point<3>()));
                tria.set_manifold(2,
                                  dealii::CylindricalManifold<3>(
                                    dealii::Tensor<1, 3>({0., 1., 0.}),
                                    dealii::Point<3>()));
              }
            else
              AssertThrow(false,
                          dealii::StandardExceptions::ExcNotImplemented());
            tria.refine_global(n_refinements_x);
          };

          const auto fu_v = [&](auto &tria) {
            if constexpr (dim_v == 3)
              {
                const double ball_R = 5.0;
                dealii::GridGenerator::hyper_ball_balanced(
                  tria, dealii::Point<dim_v>(), ball_R);
              }
            else
              AssertThrow(false,
                          dealii::StandardExceptions::ExcNotImplemented());
            tria.refine_global(n_refinements_v);
          };

          hyperdeal::GridGenerator::construct_tensor_product<dim_x, dim_v>(
            tria_x, tria_v, fu_x, fu_v);
        }

        void
        set_boundary_conditions(
          std::shared_ptr<BoundaryDescriptor<dim_x + dim_v, Number>>
            boundary_descriptor) override
        {
          boundary_descriptor->homogeneous_dirichlet_bc.insert(0);
        }

        void
        set_analytical_solution(
          std::shared_ptr<dealii::Function<dim_x + dim_v, Number>>
            &analytical_solution) override
        {
          analytical_solution.reset(new ExactSolution<dim_x + dim_v, Number>());
        }

        dealii::Tensor<1, dim_x + dim_v>
        get_transport_direction() override
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
