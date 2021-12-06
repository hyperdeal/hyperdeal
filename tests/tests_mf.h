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

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/tests/tests.h>

#include <hyper.deal/base/mpi.h>
#include <hyper.deal/matrix_free/matrix_free.h>

namespace hyperdeal
{
  struct Parameters
  {
    std::string triangulation_type;

    unsigned int degree            = 1;
    unsigned int mapping_degree    = 1;
    bool         do_collocation    = false;
    bool         do_ghost_faces    = true;
    bool         do_buffering      = false;
    bool         use_ecl           = true;
    unsigned int overlapping_level = 0;

    bool print_parameter = false;

    Parameters() = default;

    Parameters(const std::string &               file_name,
               const dealii::ConditionalOStream &pcout)
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
      prm.add_parameter("Degree", degree);
      prm.add_parameter("Verbose", print_parameter);
      prm.leave_subsection();

      prm.enter_subsection("Triangulation");
      prm.add_parameter("Type", triangulation_type);
      prm.leave_subsection();

      prm.enter_subsection("SpatialDiscretization");
      prm.add_parameter("Mapping", mapping_degree);
      prm.add_parameter("DoCollocation", do_collocation);
      prm.leave_subsection();

      prm.enter_subsection("MatrixFree");
      prm.add_parameter("GhostFaces", do_ghost_faces);
      prm.add_parameter("DoBuffering", do_buffering);
      prm.add_parameter("UseECL", use_ecl);
      prm.add_parameter("OverlappingLevel", overlapping_level);
      prm.leave_subsection();
    }
  };


  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  class MatrixFreeWrapper
  {
  public:
    static const int dim = dim_x + dim_v;

    using MF = hyperdeal::MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>;
    using VectorizedArrayTypeX = typename MF::VectorizedArrayTypeX;
    using VectorizedArrayTypeV = typename MF::VectorizedArrayTypeV;

    MatrixFreeWrapper(const MPI_Comm &   comm_global,
                      const MPI_Comm &   comm_sm,
                      const unsigned int size_x,
                      const unsigned int size_v)
      : comm_global(comm_global)
      , comm_sm(comm_sm)
      , comm_row(mpi::create_row_comm(comm_global, size_x, size_v))
      , comm_column(mpi::create_column_comm(comm_global, size_x, size_v))
      , matrix_free(comm_global, comm_sm, matrix_free_x, matrix_free_v)
    {}


    template <typename Fu>
    void
    init(const Parameters param, const Fu &create_grid)
    {
      const auto degree_x = param.degree;
      const auto degree_v = param.degree;

      const auto n_points_x = degree_x + 1;
      const auto n_points_v = degree_v + 1;

      const auto mapping_degree_x = param.mapping_degree;
      const auto mapping_degree_v = param.mapping_degree;

      // step 1: create two low-dimensional triangulations
      {
        // clang-format off
          if(param.triangulation_type == "fullydistributed")
            {
              triangulation_x.reset(new dealii::parallel::fullydistributed::Triangulation<dim_x>(comm_row));
              triangulation_v.reset(new dealii::parallel::fullydistributed::Triangulation<dim_v>(comm_column));
            }
#ifdef DEAL_II_WITH_P4EST
          else if(param.triangulation_type == "distributed")
            {
              triangulation_x.reset(new dealii::parallel::distributed::Triangulation<dim_x>(comm_row,
            dealii::Triangulation<dim_x>::none,
            dealii::parallel::distributed::Triangulation<dim_x>::construct_multigrid_hierarchy));
              triangulation_v.reset(new dealii::parallel::distributed::Triangulation<dim_v>(comm_column,
            dealii::Triangulation<dim_v>::none,
            dealii::parallel::distributed::Triangulation<dim_v>::construct_multigrid_hierarchy));
            }
#endif
          else
            AssertThrow(false, dealii::ExcMessage("Unknown triangulation!"));
        // clang-format on

        create_grid(triangulation_x, triangulation_v);
      }

      // step 2: create two low-dimensional dof-handler
      {
        dof_handler_x.reset(new dealii::DoFHandler<dim_x>(*triangulation_x));
        dof_handler_v.reset(new dealii::DoFHandler<dim_v>(*triangulation_v));

        dealii::FE_DGQ<dim_x> fe_x(degree_x);
        dealii::FE_DGQ<dim_v> fe_v(degree_v);
        dof_handler_x->distribute_dofs(fe_x);
        dof_handler_v->distribute_dofs(fe_v);
      }

      // step 3: setup two low-dimensional matrix-frees
      {
        dealii::AffineConstraints<Number> constraint;
        constraint.close();

        {
          typename dealii::MatrixFree<dim_x, Number, VectorizedArrayTypeX>::
            AdditionalData additional_data;
          additional_data.mapping_update_flags =
            dealii::update_gradients | dealii::update_JxW_values |
            dealii::update_quadrature_points;
          additional_data.mapping_update_flags_inner_faces =
            dealii::update_gradients | dealii::update_JxW_values;
          additional_data.mapping_update_flags_boundary_faces =
            dealii::update_gradients | dealii::update_JxW_values |
            dealii::update_quadrature_points;
          additional_data.hold_all_faces_to_owned_cells = true;
          additional_data.mapping_update_flags_faces_by_cells =
            dealii::update_gradients | dealii::update_JxW_values |
            dealii::update_quadrature_points;
          dealii::MappingQGeneric<dim_x> mapping_x(mapping_degree_x);

          std::vector<const dealii::DoFHandler<dim_x> *> dof_handlers{
            &*dof_handler_x};
          std::vector<const dealii::AffineConstraints<Number> *> constraints{
            &constraint};

          std::vector<dealii::Quadrature<1>> quads;
          if (param.do_collocation)
            quads.push_back(dealii::QGaussLobatto<1>(n_points_x));
          else
            quads.push_back(dealii::QGauss<1>(n_points_x));

          quads.push_back(dealii::QGauss<1>(degree_x + 1));
          quads.push_back(dealii::QGaussLobatto<1>(degree_x + 1));

          matrix_free_x.reinit(
            mapping_x, dof_handlers, constraints, quads, additional_data);
        }

        {
          typename dealii::MatrixFree<dim_v, Number, VectorizedArrayTypeV>::
            AdditionalData additional_data;
          additional_data.mapping_update_flags =
            dealii::update_gradients | dealii::update_JxW_values |
            dealii::update_quadrature_points;
          additional_data.mapping_update_flags_inner_faces =
            dealii::update_gradients | dealii::update_JxW_values;
          additional_data.mapping_update_flags_boundary_faces =
            dealii::update_gradients | dealii::update_JxW_values |
            dealii::update_quadrature_points;
          additional_data.hold_all_faces_to_owned_cells = true;
          additional_data.mapping_update_flags_faces_by_cells =
            dealii::update_gradients | dealii::update_JxW_values |
            dealii::update_quadrature_points;
          dealii::MappingQGeneric<dim_v> mapping_v(mapping_degree_v);

          std::vector<const dealii::DoFHandler<dim_v> *> dof_handlers{
            &*dof_handler_v};
          std::vector<const dealii::AffineConstraints<Number> *> constraints{
            &constraint};
          std::vector<dealii::Quadrature<1>> quads;
          if (param.do_collocation)
            quads.push_back(dealii::QGaussLobatto<1>(n_points_v));
          else
            quads.push_back(dealii::QGauss<1>(n_points_v));

          quads.push_back(dealii::QGauss<1>(degree_v + 1));
          quads.push_back(dealii::QGaussLobatto<1>(degree_v + 1));

          matrix_free_v.reinit(
            mapping_v, dof_handlers, constraints, quads, additional_data);
        }
      }

      // step 4: setup tensor-product matrixfree
      {
        typename MF::AdditionalData ad;
        ad.do_ghost_faces    = param.do_ghost_faces;
        ad.do_buffering      = param.do_buffering;
        ad.use_ecl           = param.use_ecl;
        ad.overlapping_level = param.overlapping_level;

        matrix_free.reinit(ad);
      }
    }


    const MF &
    get_matrix_free() const
    {
      return matrix_free;
    }

    dealii::types::global_dof_index
    n_dofs() const
    {
      return dof_handler_x->n_dofs() * dof_handler_v->n_dofs();
    }

    dealii::types::global_dof_index
    n_dofs_x() const
    {
      return dof_handler_x->n_dofs();
    }

    dealii::types::global_dof_index
    n_dofs_v() const
    {
      return dof_handler_v->n_dofs();
    }

    const MPI_Comm &
    get_comm_row()
    {
      return comm_row;
    }

    const MPI_Comm &
    get_comm_column()
    {
      return comm_column;
    }

  protected:
    const MPI_Comm &comm_global;
    const MPI_Comm &comm_sm;
    MPI_Comm        comm_row;
    MPI_Comm        comm_column;

    // x- and v-space objects
    std::shared_ptr<dealii::parallel::TriangulationBase<dim_x>> triangulation_x;
    std::shared_ptr<dealii::parallel::TriangulationBase<dim_v>> triangulation_v;
    std::shared_ptr<dealii::DoFHandler<dim_x>>                  dof_handler_x;
    std::shared_ptr<dealii::DoFHandler<dim_v>>                  dof_handler_v;
    dealii::MatrixFree<dim_x, Number, VectorizedArrayTypeX>     matrix_free_x;
    dealii::MatrixFree<dim_v, Number, VectorizedArrayTypeV>     matrix_free_v;
    MF                                                          matrix_free;
  };

} // namespace hyperdeal
