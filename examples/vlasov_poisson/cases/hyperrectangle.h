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

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold.h>

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
            advection[d + dim_v] = 6.0;
        }

        virtual double
        value(const dealii::Point<DIM> &p, const unsigned int = 1) const
        {
          const dealii::Tensor<1, DIM> position = p;
          double                       result =
            1.0; // std::sin(wave_number * position[0] * numbers::PI);

          for (unsigned int d = 0; d < 1; ++d)
            result = result + 0.01 * std::cos(wave_number * position[d]);
          for (unsigned int d = dim_x; d < dim_x + dim_v; ++d)
            result = result * std::exp(-0.5 * pow(position[d], 2)) /
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
          : n_subdivisions_x(3, 0)
          , n_subdivisions_v(3, 0)
        {}

        void
        add_parameters(dealii::ParameterHandler &prm) override
        {
          prm.enter_subsection("Case");

          prm.add_parameter("NRefinementsX", n_refinements_x);
          prm.add_parameter("NRefinementsV", n_refinements_v);

          prm.add_parameter("PeriodicX", do_periodic_x);
          prm.add_parameter("PeriodicV", do_periodic_v);

          prm.enter_subsection("NSubdivisionsX");
          prm.add_parameter("X", n_subdivisions_x[0]);
          prm.add_parameter("Y", n_subdivisions_x[1]);
          prm.add_parameter("Z", n_subdivisions_x[2]);
          prm.leave_subsection();

          prm.enter_subsection("NSubdivisionsV");
          prm.add_parameter("X", n_subdivisions_v[0]);
          prm.add_parameter("Y", n_subdivisions_v[1]);
          prm.add_parameter("Z", n_subdivisions_v[2]);
          prm.leave_subsection();

          prm.leave_subsection();
        }

        template <int dim_>
        void
        apply_periodicity(dealii::Triangulation<dim_> *tria,
                          const dealii::Point<dim_x> & left,
                          const dealii::Point<dim_x> & right,
                          const int                    counter = 0)
        {
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim_>::cell_iterator>>
               periodic_faces;
          auto cell = tria->begin();
          auto endc = tria->end();
          for (; cell != endc; ++cell)
            {
              for (unsigned int face_number = 0;
                   face_number < dealii::GeometryInfo<dim_>::faces_per_cell;
                   ++face_number)
                {
                  // clang-format off
                  // x-direction
                  if ((dim_ >= 1) && (std::fabs(cell->face(face_number)->center()(0) - left(0)) < 1e-12))
                    cell->face(face_number)->set_all_boundary_ids(0 + counter);
                  if ((dim_ >= 1) && (std::fabs(cell->face(face_number)->center()(0) - right(0)) < 1e-12))
                    cell->face(face_number)->set_all_boundary_ids(1 + counter);
                  // y-direction
                  if ((dim_ >= 2) && (std::fabs(cell->face(face_number)->center()(1) - left(1)) < 1e-12))
                    cell->face(face_number)->set_all_boundary_ids(2 + counter);
                  if ((dim_ >= 2) && (std::fabs(cell->face(face_number)->center()(1) - right(1)) < 1e-12))
                    cell->face(face_number)->set_all_boundary_ids(3 + counter);
                  // z-direction
                  if ((dim_ >= 3) && (std::fabs(cell->face(face_number)->center()(2) - left(2)) < 1e-12))
                    cell->face(face_number)->set_all_boundary_ids(4 + counter);
                  if ((dim_ >= 3) && (std::fabs(cell->face(face_number)->center()(2) - right(2)) < 1e-12))
                    cell->face(face_number)->set_all_boundary_ids(5 + counter);
                  // clang-format on
                }
            }

          // x-direction
          if (dim_ >= 1)
            dealii::GridTools::collect_periodic_faces(
              *tria, 0 + counter, 1 + counter, 0, periodic_faces);

          // y-direction
          if (dim_ >= 2)
            dealii::GridTools::collect_periodic_faces(
              *tria, 2 + counter, 3 + counter, 1, periodic_faces);

          // z-direction
          if (dim_ >= 3)
            dealii::GridTools::collect_periodic_faces(
              *tria, 4 + counter, 5 + counter, 2, periodic_faces);

          tria->add_periodicity(periodic_faces);
        }

        void
        create_grid(
          std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>> &tria_x,
          std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>> &tria_v)
          override
        {
          std::vector<unsigned int> repetitions_x(dim_x);
          std::vector<unsigned int> repetitions_v(dim_v);

          for (unsigned int i = 0; i < dim_x; i++)
            repetitions_x[i] = this->n_subdivisions_x[i];

          for (unsigned int i = 0; i < dim_v; i++)
            repetitions_v[i] = this->n_subdivisions_v[i];


          dealii::Point<dim_x> left_x;
          dealii::Point<dim_x> right_x;
          dealii::Point<dim_v> left_v;
          dealii::Point<dim_v> right_v;

          for (unsigned int d = 0; d < dim_x; ++d)
            left_x(d) = 0.0;

          for (unsigned int d = 0; d < dim_x; ++d)
            right_x(d) = 4.0 * dealii::numbers::PI;

          for (unsigned int d = 0; d < dim_v; ++d)
            left_v(d) = -6.0;

          for (unsigned int d = 0; d < dim_v; ++d)
            right_v(d) = 6.0;

          if (auto triangulation_x = dynamic_cast<
                dealii::parallel::distributed::Triangulation<dim_x> *>(
                &*tria_x))
            {
              if (auto triangulation_v = dynamic_cast<
                    dealii::parallel::distributed::Triangulation<dim_v> *>(
                    &*tria_v))
                {
                  dealii::GridGenerator::subdivided_hyper_rectangle(
                    *triangulation_x, repetitions_x, left_x, right_x);
                  dealii::GridGenerator::subdivided_hyper_rectangle(
                    *triangulation_v, repetitions_v, left_v, right_v);

                  if (do_periodic_x)
                    apply_periodicity(triangulation_x, left_x, right_x);
                  if (do_periodic_v)
                    apply_periodicity(triangulation_v,
                                      left_v,
                                      right_v,
                                      2 * dim_x);

                  triangulation_x->refine_global(n_refinements_x);
                  triangulation_v->refine_global(n_refinements_v);
                }
              else
                AssertThrow(false,
                            dealii::ExcMessage("Unknown triangulation!"));
            }
          else if (auto triangulation_x =
                     dynamic_cast<dealii::parallel::fullydistributed::
                                    Triangulation<dim_x> *>(&*tria_x))
            {
              if (auto triangulation_v = dynamic_cast<
                    dealii::parallel::fullydistributed::Triangulation<dim_v> *>(
                    &*tria_v))
                {
                  {
                    auto comm = tria_x->get_communicator();
                    dealii::Triangulation<dim_x> tria;
                    dealii::GridGenerator::subdivided_hyper_rectangle(
                      tria, repetitions_x, left_x, right_x);

                    if (do_periodic_x)
                      apply_periodicity(&tria, left_x, right_x);
                    tria.refine_global(n_refinements_x);
                    dealii::GridTools::partition_triangulation_zorder(
                      dealii::Utilities::MPI::n_mpi_processes(comm),
                      tria,
                      false);
                    dealii::GridTools::partition_multigrid_levels(tria);

                    const auto construction_data =
                      dealii::TriangulationDescription::Utilities::
                        create_description_from_triangulation(tria, comm, true);
                    triangulation_x->create_triangulation(construction_data);
                  }
                  if (do_periodic_x)
                    apply_periodicity(
                      dynamic_cast<dealii::Triangulation<dim_x> *>(&*tria_x),
                      left_x,
                      right_x,
                      20);

                  {
                    auto comm = tria_v->get_communicator();
                    dealii::Triangulation<dim_v> tria;
                    dealii::GridGenerator::subdivided_hyper_rectangle(
                      tria, repetitions_v, left_v, right_v);

                    if (do_periodic_v)
                      apply_periodicity(&tria, left_v, right_v, 2 * dim_x);
                    tria.refine_global(n_refinements_v);
                    dealii::GridTools::partition_triangulation_zorder(
                      dealii::Utilities::MPI::n_mpi_processes(comm),
                      tria,
                      false);
                    dealii::GridTools::partition_multigrid_levels(tria);

                    const auto construction_data =
                      dealii::TriangulationDescription::Utilities::
                        create_description_from_triangulation(tria, comm, true);
                    triangulation_v->create_triangulation(construction_data);
                  }
                  if (do_periodic_v)
                    apply_periodicity(
                      dynamic_cast<dealii::Triangulation<dim_v> *>(&*tria_v),
                      left_v,
                      right_v,
                      20 + 2 * dim_x);
                }
              else
                AssertThrow(false,
                            dealii::ExcMessage("Unknown triangulation!"));
            }
          else
            AssertThrow(false, dealii::ExcMessage("Unknown triangulation!"));
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

      private:
        unsigned int n_refinements_x = 0;
        unsigned int n_refinements_v = 0;
        bool         do_periodic_x   = true;
        bool         do_periodic_v   = true;

        std::vector<unsigned int> n_subdivisions_x;
        std::vector<unsigned int> n_subdivisions_v;
      };
    } // namespace hyperrectangle
  }   // namespace vp
} // namespace hyperdeal

#endif
