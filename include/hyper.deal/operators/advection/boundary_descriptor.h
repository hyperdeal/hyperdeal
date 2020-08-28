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

#ifndef HYPERDEAL_USERINTERFACE_BOUNDARYDESCRIPTOR
#define HYPERDEAL_USERINTERFACE_BOUNDARYDESCRIPTOR

#include <hyper.deal/base/config.h>

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

namespace hyperdeal
{
  namespace advection
  {
    /**
     * Type of the boundary.
     */
    enum class BoundaryType
    {
      Undefined,
      DirichletInhomogenous,
      DirichletHomogenous,
    };

    /**
     * Class managing different types of boundary conditions.
     */
    template <int dim, typename Number>
    struct BoundaryDescriptor
    {
      /**
       * Dirichlet boundaries.
       */
      std::map<dealii::types::boundary_id,
               std::shared_ptr<dealii::Function<dim, Number>>>
        dirichlet_bc;

      std::set<dealii::types::boundary_id> homogeneous_dirichlet_bc;

      /**
       * Return boundary type.
       */
      inline DEAL_II_ALWAYS_INLINE //
        BoundaryType
        get_boundary_type(dealii::types::boundary_id const &boundary_id) const
      {
        if (this->dirichlet_bc.find(boundary_id) != this->dirichlet_bc.end())
          return BoundaryType::DirichletInhomogenous;

        if (this->homogeneous_dirichlet_bc.find(boundary_id) !=
            this->homogeneous_dirichlet_bc.end())
          return BoundaryType::DirichletInhomogenous;

        AssertThrow(false,
                    dealii::ExcMessage(
                      "Boundary type of face is invalid or not implemented."));

        return BoundaryType::Undefined;
      }

      /**
       * Return boundary type and the associated function.
       */
      inline DEAL_II_ALWAYS_INLINE //
        std::pair<BoundaryType, std::shared_ptr<dealii::Function<dim, Number>>>
        get_boundary(dealii::types::boundary_id const &boundary_id) const
      {
        // process inhomogeneous Dirichlet BC
        {
          auto res = this->dirichlet_bc.find(boundary_id);
          if (res != this->dirichlet_bc.end())
            return {BoundaryType::DirichletInhomogenous, res->second};
        }

        // process homogeneous Dirichlet BC
        {
          auto res = this->homogeneous_dirichlet_bc.find(boundary_id);
          if (res != this->homogeneous_dirichlet_bc.end())
            return {BoundaryType::DirichletHomogenous,
                    std::shared_ptr<dealii::Function<dim, Number>>(
                      new dealii::Functions::ZeroFunction<dim, Number>())};
        }

        AssertThrow(false,
                    dealii::ExcMessage(
                      "Boundary type of face is invalid or not implemented."));

        return {BoundaryType::Undefined,
                std::shared_ptr<dealii::Function<dim>>(
                  new dealii::Functions::ZeroFunction<dim, Number>())};
      }

      /**
       * Set time for all internal functions.
       */
      void
      set_time(const Number time)
      {
        for (auto &bc : dirichlet_bc)
          bc.second->set_time(time);
      }
    };

  } // namespace advection
} // namespace hyperdeal

#endif
