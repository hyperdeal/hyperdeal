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
  namespace vp
  {
    namespace torus_hyperball
    {
      static const double torus_R = 6.2; // ITER
      static const double torus_r = 2.0;
      static const double ball_R  = 5.0;

      template <int dim_x, int dim_v, typename Number = double>
      class ExactSolution : public dealii::Function<dim_x + dim_v, Number>
      {
      public:
        ExactSolution(const double time = 0.)
          : dealii::Function<dim_x + dim_v, Number>(1, time)
          , wave_number(0.5)
        {
          for (unsigned int d = 0; d < dim_x; ++d)
            advection[d] = 1.0;
          for (unsigned int d = 0; d < dim_v; ++d)
            advection[d + dim_v] = 6.0;
        }

        virtual double
        value(const dealii::Point<dim_x + dim_v> &p,
              const unsigned int = 1) const override
        {
          // project p on x-z plane -> direction vector
          const dealii::Point<dim_x> pp(p[0], 0.0, p[2]);

          // normalized and scale direction vector, s.t, its length equals to R
          // and subtract from point -> distance from the center of the tube
          const double distance = (torus_R * (pp / pp.norm()) -
                                   dealii::Point<dim_x>(p[0], p[1], p[2]))
                                    .norm();

          double result = std::exp(-1.0 * pow(distance, 2));

          for (unsigned int d = dim_x; d < dim_x + dim_v; ++d)
            result = result * std::exp(-0.5 * pow(p[d], 2)) /
                     std::sqrt(2.0 * dealii::numbers::PI);

          return result;
        }

        dealii::Tensor<1, dim_x + dim_v>
        get_transport_direction() const
        {
          return advection;
        }

      private:
        dealii::Tensor<1, dim_x + dim_v> advection;
        const double                     wave_number;
      };

      template <int dim_x, int dim_v, int degree, typename Number>
      class Initializer
        : public hyperdeal::vp::Initializer<dim_x, dim_v, degree, Number>
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
              dealii::GridGenerator::hyper_ball_balanced(tria,
                                                         dealii::Point<dim_v>(),
                                                         ball_R);
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
          std::shared_ptr<
            hyperdeal::advection::BoundaryDescriptor<dim_x + dim_v, Number>>
            boundary_descriptor) override
        {
          boundary_descriptor->homogeneous_dirichlet_bc.insert(0);
        }

        void
        set_analytical_solution(
          std::shared_ptr<dealii::Function<dim_x + dim_v, Number>>
            &analytical_solution) override
        {
          analytical_solution.reset(new ExactSolution<dim_x, dim_v, Number>());
        }

        dealii::Tensor<1, dim_x + dim_v>
        get_transport_direction() override
        {
          return ExactSolution<dim_x, dim_v, Number>()
            .get_transport_direction();
        }

        LaplaceOperatorBCType
        get_poisson_problem_bc_type() const override
        {
          return LaplaceOperatorBCType::NBC;
        }

      private:
        unsigned int n_refinements_x = 0;
        unsigned int n_refinements_v = 0;
      };
    } // namespace torus_hyperball
  }   // namespace vp
} // namespace hyperdeal

#endif
