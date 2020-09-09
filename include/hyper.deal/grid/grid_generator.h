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

#ifndef HYPERDEAL_GRID_GRIDGENERATOR
#define HYPERDEAL_GRID_GRIDGENERATOR

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>

namespace hyperdeal
{
  /**
   * This namespace provides a collection of functions for generating
   * triangulations for some basic tensor-product geometries.
   *
   * TODO: replace shared_ptr!
   */
  namespace GridGenerator
  {
    /**
     * Create two dealii::GridGenerator::subdivided_hyper_rectangle().
     */
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
      const bool                       do_periodic_v,
      const bool                       with_internal_deformation = false);


    /**
     * Create two dealii::GridGenerator::hyper_cube().
     */
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
      const bool          do_periodic_v);

    /**
     * Same as above but that the parameters of x- and v-space triangulation are
     * chosen the same way.
     */
    template <int dim_x, int dim_v>
    void
    hyper_cube(
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>> &tria_x,
      std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>> &tria_v,
      const bool                                                   do_periodic,
      const unsigned int &n_refinements,
      const double        left  = 0.0,
      const double        right = 1.0);

    /**
     * Same as above but that each coarse cell is subdivided in z-direction.
     * In contrast to hyperdeal::GridGenerator::subdivided_hyper_rectangle,
     * one can give the coarse cells different orientations
     * (0 &ge; orientation_x, orientation_v &lt; 16). This function is
     * particular useful, since it is the most simple prototype of an
     * unstrucured grid and enables straight-forward debugging.
     *
     * @note Only implemented for dim_x==dim_v==3.
     */
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
      const unsigned int &        orientation_v);

    /**
     * Create two dealii::GridGenerator::hyper_ball().
     *
     * @note Before refinement, we remove the manifolds from geometric entities
     *   of the triangluations so that the final bounding faces look exactly
     *   like in the case of hyperdeal::GridGenerator::hyper_cube().
     */
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
      const bool                  do_periodic_v);

  } // namespace GridGenerator

} // namespace hyperdeal

#endif