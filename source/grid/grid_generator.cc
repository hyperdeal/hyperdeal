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

#include <hyper.deal/grid/grid_generator.h>

namespace hyperdeal
{
  /**
   * TODO: replace shared_ptr!
   */
  namespace GridGenerator
  {
    namespace internal
    {
      template <int dim_>
      void
      apply_periodicity(dealii::Triangulation<dim_> *tria,
                        const dealii::Point<dim_> &  left,
                        const dealii::Point<dim_> &  right,
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
                // x-direction
                if ((dim_ >= 1) &&
                    (std::fabs(cell->face(face_number)->center()(0) - left[0]) <
                     1e-12))
                  cell->face(face_number)->set_all_boundary_ids(0 + counter);
                if ((dim_ >= 1) &&
                    (std::fabs(cell->face(face_number)->center()(0) -
                               right[0]) < 1e-12))
                  cell->face(face_number)->set_all_boundary_ids(1 + counter);
                // y-direction
                if ((dim_ >= 2) &&
                    (std::fabs(cell->face(face_number)->center()(1) - left[1]) <
                     1e-12))
                  cell->face(face_number)->set_all_boundary_ids(2 + counter);
                if ((dim_ >= 2) &&
                    (std::fabs(cell->face(face_number)->center()(1) -
                               right[1]) < 1e-12))
                  cell->face(face_number)->set_all_boundary_ids(3 + counter);
                // z-direction
                if ((dim_ >= 3) &&
                    (std::fabs(cell->face(face_number)->center()(2) - left[2]) <
                     1e-12))
                  cell->face(face_number)->set_all_boundary_ids(4 + counter);
                if ((dim_ >= 3) &&
                    (std::fabs(cell->face(face_number)->center()(2) -
                               right[2]) < 1e-12))
                  cell->face(face_number)->set_all_boundary_ids(5 + counter);
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

      template <int dim_>
      void
      apply_periodicity(dealii::Triangulation<dim_> *tria,
                        const int                    counter = 0,
                        const double                 left    = 0.0,
                        const double                 right   = 1.0)
      {
        // clang-format off
        const dealii::Point< dim_ > point_left = dim_ == 1 ? dealii::Point< dim_ >(left) : (dim_ == 2 ? dealii::Point< dim_ >(left, left) : dealii::Point< dim_ >(left, left, left)); 
        const dealii::Point< dim_ > point_right = dim_ == 1 ? dealii::Point< dim_ >(right) : (dim_ == 2 ? dealii::Point< dim_ >(right, right) : dealii::Point< dim_ >(right, right, right));
        // clang-format on

        apply_periodicity(tria, counter, point_left, point_right);
      }
    } // namespace internal


    template <int dim_x, int dim_v>
    void
    subdivided_hyper_rectangle(
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>> &tria_x,
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>> &tria_v,
      const unsigned int &             n_refinements_x,
      const std::vector<unsigned int> &repetitions_x,
      const dealii::Point<dim_x> &     left_x,
      const dealii::Point<dim_x> &     right_x,
      const bool                       do_periodic_x,
      const unsigned int &             n_refinements_v,
      const std::vector<unsigned int> &repetitions_v,
      const dealii::Point<dim_v> &     left_v,
      const dealii::Point<dim_v> &     right_v,
      const bool                       do_periodic_v)
    {
      if (auto triangulation_x =
            dynamic_cast<dealii::parallel::distributed::Triangulation<dim_x> *>(
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
                internal::apply_periodicity(triangulation_x, left_x, right_x);
              if (do_periodic_v)
                internal::apply_periodicity(triangulation_v,
                                            left_v,
                                            right_v,
                                            2 * dim_x);

              triangulation_x->refine_global(n_refinements_x);
              triangulation_v->refine_global(n_refinements_v);
            }
          else
            AssertThrow(false, dealii::ExcMessage("Unknown triangulation!"));
        }
      else if (auto triangulation_x = dynamic_cast<
                 dealii::parallel::fullydistributed::Triangulation<dim_x> *>(
                 &*tria_x))
        {
          if (auto triangulation_v = dynamic_cast<
                dealii::parallel::fullydistributed::Triangulation<dim_v> *>(
                &*tria_v))
            {
              {
                auto                         comm = tria_x->get_communicator();
                dealii::Triangulation<dim_x> tria(
                  dealii::Triangulation<
                    dim_x>::limit_level_difference_at_vertices);
                dealii::GridGenerator::subdivided_hyper_rectangle(tria,
                                                                  repetitions_x,
                                                                  left_x,
                                                                  right_x);

                if (do_periodic_x)
                  internal::apply_periodicity(&tria, left_x, right_x);
                tria.refine_global(n_refinements_x);
                dealii::GridTools::partition_triangulation_zorder(
                  dealii::Utilities::MPI::n_mpi_processes(comm), tria, false);
                dealii::GridTools::partition_multigrid_levels(tria);

                const auto construction_data =
                  dealii::TriangulationDescription::Utilities::
                    create_description_from_triangulation(
                      tria,
                      comm,
                      dealii::TriangulationDescription::Settings::
                        construct_multigrid_hierarchy);
                triangulation_x->create_triangulation(construction_data);
              }
              if (do_periodic_x)
                internal::apply_periodicity(
                  dynamic_cast<dealii::Triangulation<dim_x> *>(&*tria_x),
                  left_x,
                  right_x,
                  20);

              {
                auto                         comm = tria_v->get_communicator();
                dealii::Triangulation<dim_v> tria(
                  dealii::Triangulation<
                    dim_v>::limit_level_difference_at_vertices);
                dealii::GridGenerator::subdivided_hyper_rectangle(tria,
                                                                  repetitions_v,
                                                                  left_v,
                                                                  right_v);

                if (do_periodic_v)
                  internal::apply_periodicity(&tria,
                                              left_v,
                                              right_v,
                                              2 * dim_x);
                tria.refine_global(n_refinements_v);
                dealii::GridTools::partition_triangulation_zorder(
                  dealii::Utilities::MPI::n_mpi_processes(comm), tria, false);
                dealii::GridTools::partition_multigrid_levels(tria);

                const auto construction_data =
                  dealii::TriangulationDescription::Utilities::
                    create_description_from_triangulation(
                      tria,
                      comm,
                      dealii::TriangulationDescription::Settings::
                        construct_multigrid_hierarchy);
                triangulation_v->create_triangulation(construction_data);
              }
              if (do_periodic_v)
                internal::apply_periodicity(
                  dynamic_cast<dealii::Triangulation<dim_v> *>(&*tria_v),
                  left_v,
                  right_v,
                  20 + 2 * dim_x);
            }
          else
            AssertThrow(false, dealii::ExcMessage("Unknown triangulation!"));
        }
      else
        AssertThrow(false, dealii::ExcMessage("Unknown triangulation!"));
    }


    template <int dim_x, int dim_v>
    void
    hyper_cube(
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>> &tria_x,
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>> &tria_v,
      const unsigned int &n_refinements_x,
      const double        left_x,
      const double        right_x,
      const bool          do_periodic_x,
      const unsigned int &n_refinements_v,
      const double        left_v,
      const double        right_v,
      const bool          do_periodic_v)
    {
      // clang-format off
      const dealii::Point< dim_x > p_left_x = dim_x == 1 ? dealii::Point< dim_x >(left_x) : (dim_x == 2 ? dealii::Point< dim_x >(left_x, left_x) : dealii::Point< dim_x >(left_x, left_x, left_x)); 
      const dealii::Point< dim_x > p_right_x = dim_x == 1 ? dealii::Point< dim_x >(right_x) : (dim_x == 2 ? dealii::Point< dim_x >(right_x, right_x) : dealii::Point< dim_x >(right_x, right_x, right_x));
      const dealii::Point< dim_v > p_left_v = dim_v == 1 ? dealii::Point< dim_v >(left_v) : (dim_v == 2 ? dealii::Point< dim_v >(left_v, left_v) : dealii::Point< dim_v >(left_v, left_v, left_v)); 
      const dealii::Point< dim_v > p_right_v = dim_v == 1 ? dealii::Point< dim_v >(right_v) : (dim_v == 2 ? dealii::Point< dim_v >(right_v, right_v) : dealii::Point< dim_v >(right_v, right_v, right_v));
      
      std::vector<unsigned int> repetitions_x(dim_x, 1);
      std::vector<unsigned int> repetitions_v(dim_v, 1);
      
      subdivided_hyper_rectangle(tria_x, tria_v, 
        n_refinements_x, repetitions_x, p_left_x, p_right_x, do_periodic_x,
        n_refinements_v, repetitions_v, p_left_v, p_right_v, do_periodic_v);
      // clang-format on
    }

    template <int dim_x, int dim_v>
    void
    hyper_cube(
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>> &tria_x,
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>> &tria_v,
      const bool                                                   do_periodic,
      const unsigned int &n_refinements,
      const double        left,
      const double        right)
    {
      // clang-format off
      
      hyper_cube(tria_x, tria_v, 
        n_refinements, left, right, do_periodic,
        n_refinements, left, right, do_periodic);
      // clang-format on
    }

    template <int dim_x, int dim_v>
    void
    subdivided_hyper_ball(
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>> &tria_x,
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>> &tria_v,
      const unsigned int &        n_refinements_x,
      const dealii::Point<dim_x> &left_x,
      const dealii::Point<dim_x> &right_x,
      const bool                  do_periodic_x,
      const unsigned int &        n_refinements_v,
      const dealii::Point<dim_v> &left_v,
      const dealii::Point<dim_v> &right_v,
      const bool                  do_periodic_v)
    {
      if (auto triangulation_x =
            dynamic_cast<dealii::parallel::distributed::Triangulation<dim_x> *>(
              &*tria_x))
        {
          if (auto triangulation_v = dynamic_cast<
                dealii::parallel::distributed::Triangulation<dim_v> *>(
                &*tria_v))
            {
              dealii::GridGenerator::hyper_ball(
                *triangulation_x,
                dim_x == 2 ? dealii::Point<dim_x>(0.0, 0.0) :
                             dealii::Point<dim_x>(0.0, 0.0, 0.0),
                2.0 * (dim_x == 2 ? std::sqrt(0.5) : std::sqrt(0.75)));

              for (auto &cell : *triangulation_x)
                cell.set_all_manifold_ids(dealii::numbers::flat_manifold_id);
              ;
              dealii::GridGenerator::hyper_ball(
                *triangulation_v,
                dim_v == 2 ? dealii::Point<dim_v>(0.0, 0.0) :
                             dealii::Point<dim_v>(0.0, 0.0, 0.0),
                2.0 * (dim_v == 2 ? std::sqrt(0.5) : std::sqrt(0.75)));

              for (auto &cell : *triangulation_v)
                cell.set_all_manifold_ids(dealii::numbers::flat_manifold_id);

              if (do_periodic_x)
                internal::apply_periodicity(triangulation_x, left_x, right_x);
              if (do_periodic_v)
                internal::apply_periodicity(triangulation_v,
                                            left_v,
                                            right_v,
                                            2 * dim_x);

              triangulation_x->refine_global(n_refinements_x);
              triangulation_v->refine_global(n_refinements_v);
            }
          else
            AssertThrow(false, dealii::ExcMessage("Unknown triangulation!"));
        }
      else if (auto triangulation_x = dynamic_cast<
                 dealii::parallel::fullydistributed::Triangulation<dim_x> *>(
                 &*tria_x))
        {
          if (auto triangulation_v = dynamic_cast<
                dealii::parallel::fullydistributed::Triangulation<dim_v> *>(
                &*tria_v))
            {
              {
                auto                         comm = tria_x->get_communicator();
                dealii::Triangulation<dim_x> tria(
                  dealii::Triangulation<
                    dim_x>::limit_level_difference_at_vertices);

                dealii::GridGenerator::hyper_ball(
                  tria,
                  dim_x == 2 ? dealii::Point<dim_x>(0.0, 0.0) :
                               dealii::Point<dim_x>(0.0, 0.0, 0.0),
                  2.0 * (dim_x == 2 ? std::sqrt(0.5) : std::sqrt(0.75)));

                for (auto &cell : tria)
                  cell.set_all_manifold_ids(dealii::numbers::flat_manifold_id);

                if (do_periodic_x)
                  internal::apply_periodicity(&tria, left_x, right_x);
                tria.refine_global(n_refinements_x);
                dealii::GridTools::partition_triangulation_zorder(
                  dealii::Utilities::MPI::n_mpi_processes(comm), tria, false);
                dealii::GridTools::partition_multigrid_levels(tria);

                const auto construction_data =
                  dealii::TriangulationDescription::Utilities::
                    create_description_from_triangulation(
                      tria,
                      comm,
                      dealii::TriangulationDescription::Settings::
                        construct_multigrid_hierarchy);
                triangulation_x->create_triangulation(construction_data);
              }
              if (do_periodic_x)
                internal::apply_periodicity(
                  dynamic_cast<dealii::Triangulation<dim_x> *>(&*tria_x),
                  left_x,
                  right_x,
                  20);

              {
                auto                         comm = tria_v->get_communicator();
                dealii::Triangulation<dim_v> tria(
                  dealii::Triangulation<
                    dim_v>::limit_level_difference_at_vertices);

                ;
                dealii::GridGenerator::hyper_ball(
                  tria,
                  dim_v == 2 ? dealii::Point<dim_v>(0.0, 0.0) :
                               dealii::Point<dim_v>(0.0, 0.0, 0.0),
                  2.0 * (dim_v == 2 ? std::sqrt(0.5) : std::sqrt(0.75)));

                for (auto &cell : tria)
                  cell.set_all_manifold_ids(dealii::numbers::flat_manifold_id);

                if (do_periodic_v)
                  internal::apply_periodicity(&tria,
                                              left_v,
                                              right_v,
                                              2 * dim_x);
                tria.refine_global(n_refinements_v);
                dealii::GridTools::partition_triangulation_zorder(
                  dealii::Utilities::MPI::n_mpi_processes(comm), tria, false);
                dealii::GridTools::partition_multigrid_levels(tria);

                const auto construction_data =
                  dealii::TriangulationDescription::Utilities::
                    create_description_from_triangulation(
                      tria,
                      comm,
                      dealii::TriangulationDescription::Settings::
                        construct_multigrid_hierarchy);
                triangulation_v->create_triangulation(construction_data);
              }
              if (do_periodic_v)
                internal::apply_periodicity(
                  dynamic_cast<dealii::Triangulation<dim_v> *>(&*tria_v),
                  left_v,
                  right_v,
                  20 + 2 * dim_x);
            }
          else
            AssertThrow(false, dealii::ExcMessage("Unknown triangulation!"));
        }
      else
        AssertThrow(false, dealii::ExcMessage("Unknown triangulation!"));
    }



    template <int dim>
    void
    orientated_hyper_cube_impl(dealii::Triangulation<dim> &, int)
    {
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
    }



    template <>
    void orientated_hyper_cube_impl(dealii::Triangulation<3> &triangulation,
                                    int                       orientation)
    {
      AssertIndexRange(orientation, 16);

      dealii::Point<3> vertices_1[] = {dealii::Point<3>(-1., -1., -1.),
                                       dealii::Point<3>(+1., -1., -1.),
                                       dealii::Point<3>(-1., +1., -1.),
                                       dealii::Point<3>(+1., +1., -1.),
                                       dealii::Point<3>(-1., -1., +0.),
                                       dealii::Point<3>(+1., -1., +0.),
                                       dealii::Point<3>(-1., +1., +0.),
                                       dealii::Point<3>(+1., +1., +0.),
                                       dealii::Point<3>(-1., -1., +1.),
                                       dealii::Point<3>(+1., -1., +1.),
                                       dealii::Point<3>(-1., +1., +1.),
                                       dealii::Point<3>(+1., +1., +1.)};
      std::vector<dealii::Point<3>> vertices(&vertices_1[0], &vertices_1[12]);

      std::vector<dealii::CellData<3>> cells(2, dealii::CellData<3>());

      /* cell 0 */
      int cell_vertices_0[dealii::GeometryInfo<3>::vertices_per_cell] = {
        0, 1, 2, 3, 4, 5, 6, 7};

      /* cell 1 */
      int cell_vertices_1[8][dealii::GeometryInfo<3>::vertices_per_cell] = {
        {4, 5, 6, 7, 8, 9, 10, 11},
        {5, 7, 4, 6, 9, 11, 8, 10},
        {7, 6, 5, 4, 11, 10, 9, 8},
        {6, 4, 7, 5, 10, 8, 11, 9},
        {9, 8, 11, 10, 5, 4, 7, 6},
        {8, 10, 9, 11, 4, 6, 5, 7},
        {10, 11, 8, 9, 6, 7, 4, 5},
        {11, 9, 10, 8, 7, 5, 6, 4}};

      for (const unsigned int j : dealii::GeometryInfo<3>::vertex_indices())
        {
          cells[orientation < 8 ? 0 : 1].vertices[j] = cell_vertices_0[j];
          cells[orientation < 8 ? 1 : 0].vertices[j] =
            cell_vertices_1[orientation % 8][j];
        }


      triangulation.create_triangulation(vertices,
                                         cells,
                                         dealii::SubCellData());
    }



    template <int dim_x, int dim_v>
    void
    orientated_hyper_cube(
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>> &tria_x,
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>> &tria_v,
      const unsigned int &        n_refinements_x,
      const dealii::Point<dim_x> &left_x,
      const dealii::Point<dim_x> &right_x,
      const bool                  do_periodic_x,
      const unsigned int &        orientation_x,
      const unsigned int &        n_refinements_v,
      const dealii::Point<dim_v> &left_v,
      const dealii::Point<dim_v> &right_v,
      const bool                  do_periodic_v,
      const unsigned int &        orientation_v)
    {
      if (auto triangulation_x =
            dynamic_cast<dealii::parallel::distributed::Triangulation<dim_x> *>(
              &*tria_x))
        {
          if (auto triangulation_v = dynamic_cast<
                dealii::parallel::distributed::Triangulation<dim_v> *>(
                &*tria_v))
            {
              orientated_hyper_cube_impl(*triangulation_x, orientation_x);

              orientated_hyper_cube_impl(*triangulation_v, orientation_v);

              if (do_periodic_x)
                internal::apply_periodicity(triangulation_x, left_x, right_x);
              if (do_periodic_v)
                internal::apply_periodicity(triangulation_v,
                                            left_v,
                                            right_v,
                                            2 * dim_x);

              triangulation_x->refine_global(n_refinements_x);
              triangulation_v->refine_global(n_refinements_v);
            }
          else
            AssertThrow(false, dealii::ExcMessage("Unknown triangulation!"));
        }
      else if (auto triangulation_x = dynamic_cast<
                 dealii::parallel::fullydistributed::Triangulation<dim_x> *>(
                 &*tria_x))
        {
          if (auto triangulation_v = dynamic_cast<
                dealii::parallel::fullydistributed::Triangulation<dim_v> *>(
                &*tria_v))
            {
              {
                auto                         comm = tria_x->get_communicator();
                dealii::Triangulation<dim_x> tria(
                  dealii::Triangulation<
                    dim_x>::limit_level_difference_at_vertices);

                orientated_hyper_cube_impl(tria, orientation_x);

                if (do_periodic_x)
                  internal::apply_periodicity(&tria, left_x, right_x);
                tria.refine_global(n_refinements_x);
                dealii::GridTools::partition_triangulation_zorder(
                  dealii::Utilities::MPI::n_mpi_processes(comm), tria, false);
                dealii::GridTools::partition_multigrid_levels(tria);

                const auto construction_data =
                  dealii::TriangulationDescription::Utilities::
                    create_description_from_triangulation(
                      tria,
                      comm,
                      dealii::TriangulationDescription::Settings::
                        construct_multigrid_hierarchy);
                triangulation_x->create_triangulation(construction_data);
              }
              if (do_periodic_x)
                internal::apply_periodicity(
                  dynamic_cast<dealii::Triangulation<dim_x> *>(&*tria_x),
                  left_x,
                  right_x,
                  20);

              {
                auto                         comm = tria_v->get_communicator();
                dealii::Triangulation<dim_v> tria(
                  dealii::Triangulation<
                    dim_v>::limit_level_difference_at_vertices);

                orientated_hyper_cube_impl(tria, orientation_v);

                for (auto &cell : tria)
                  cell.set_all_manifold_ids(dealii::numbers::flat_manifold_id);

                if (do_periodic_v)
                  internal::apply_periodicity(&tria,
                                              left_v,
                                              right_v,
                                              2 * dim_x);
                tria.refine_global(n_refinements_v);
                dealii::GridTools::partition_triangulation_zorder(
                  dealii::Utilities::MPI::n_mpi_processes(comm), tria, false);
                dealii::GridTools::partition_multigrid_levels(tria);

                const auto construction_data =
                  dealii::TriangulationDescription::Utilities::
                    create_description_from_triangulation(
                      tria,
                      comm,
                      dealii::TriangulationDescription::Settings::
                        construct_multigrid_hierarchy);
                triangulation_v->create_triangulation(construction_data);
              }
              if (do_periodic_v)
                internal::apply_periodicity(
                  dynamic_cast<dealii::Triangulation<dim_v> *>(&*tria_v),
                  left_v,
                  right_v,
                  20 + 2 * dim_x);
            }
          else
            AssertThrow(false, dealii::ExcMessage("Unknown triangulation!"));
        }
      else
        AssertThrow(false, dealii::ExcMessage("Unknown triangulation!"));
    }

#include "grid_generator.inst"

  } // namespace GridGenerator

} // namespace hyperdeal
