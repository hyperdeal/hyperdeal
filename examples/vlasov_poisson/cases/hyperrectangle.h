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
  namespace vp
  {
    namespace hyperrectangle
    {
      template <int dim_x, int dim_v, typename Number = double>
      class ExactSolution : public dealii::Function<dim_x + dim_v, Number>
      {
      public:
        static const unsigned int DIM = dim_x + dim_v;
        ExactSolution(const double time = 0.)
          : dealii::Function<DIM, Number>(1, time)
          , wave_number(0.5)
        {
          for (unsigned int d = 0; d < dim_x; ++d)
            advection[d] = 1.0;
          for (unsigned int d = 0; d < dim_v; ++d)
            advection[d + dim_v] = 5.0;
        }

        virtual double
        value(const dealii::Point<DIM> &p,
              const unsigned int = 1) const override
        {
          dealii::Tensor<1, DIM> position = p;
          position[1] -= 1.0;
          double result =
            1.0; // std::sin(wave_number * position[0] * numbers::PI);

          for (unsigned int d = 0; d < 1; ++d)
            result = result + 0.01 * std::cos(wave_number * position[d]);
          for (unsigned int d = dim_x; d < dim_x + dim_v; ++d)
            result = result * pow(position[d], 2) *
                     std::exp(-0.5 * pow(position[d], 2)) /
                     std::sqrt(2.0 * dealii::numbers::PI);

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
        : public hyperdeal::vp::Initializer<dim_x, dim_v, degree, Number>
      {
      public:
        Initializer()
          : n_subdivisions_x(dim_x, 4)
          , n_subdivisions_v(dim_v, 4)
        {}

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

          prm.add_parameter(
            "DeformMesh",
            deform_mesh,
            "Deform Cartesian mesh with an arbitrary non-linear manifold.");

          prm.leave_subsection();
        }

        void
        create_grid(
          std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>> &tria_x,
          std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>> &tria_v)
          override
        {
          dealii::Point<dim_x> p1_x;
          for (unsigned int d = 0; d < dim_x; ++d)
            p1_x(d) = 0.0;
          dealii::Point<dim_x> p2_x;
          for (unsigned int d = 0; d < dim_x; ++d)
            p2_x(d) = 4.0 * dealii::numbers::PI;
          dealii::Point<dim_v> p1_v;
          for (unsigned int d = 0; d < dim_v; ++d)
            p1_v(d) = -5.0;
          dealii::Point<dim_v> p2_v;
          for (unsigned int d = 0; d < dim_v; ++d)
            p2_v(d) = 5.0;

          // clang-format off
          hyperdeal::GridGenerator::subdivided_hyper_rectangle(tria_x, tria_v, 
              n_refinements_x, n_subdivisions_x, p1_x, p2_x, do_periodic_x,
              n_refinements_v, n_subdivisions_v, p1_v, p2_v, do_periodic_v, deform_mesh);
          // clang-format on
        }

        void
        set_boundary_conditions(
          std::shared_ptr<
            hyperdeal::advection::BoundaryDescriptor<dim_x + dim_v, Number>>
            boundary_descriptor) override
        {
          boundary_descriptor->dirichlet_bc[0] =
            std::shared_ptr<dealii::Function<dim_x + dim_v, Number>>(
              new ExactSolution<dim_x, dim_v, Number>());

          // TODO: hack for 1D
          boundary_descriptor->dirichlet_bc[1] =
            std::shared_ptr<dealii::Function<dim_x + dim_v, Number>>(
              new ExactSolution<dim_x, dim_v, Number>());
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
          return LaplaceOperatorBCType::PBC;
        }

      private:
        unsigned int n_refinements_x = 0;
        unsigned int n_refinements_v = 0;
        bool         do_periodic_x   = true;
        bool         do_periodic_v   = true;

        std::vector<unsigned int> n_subdivisions_x;
        std::vector<unsigned int> n_subdivisions_v;

        bool deform_mesh = false;
      };
    } // namespace hyperrectangle
  }   // namespace vp
} // namespace hyperdeal

#endif
