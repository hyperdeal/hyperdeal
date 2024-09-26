#include <deal.II/base/bounding_box.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_compute_index_owner_internal.h>
#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/particles/data_out.h>
#include <deal.II/particles/particle_handler.h>

using namespace dealii;

/**
 * MPI utility functions from the hyper.deal library.
 */
namespace dealii
{
  namespace Utilities
  {
    namespace MPI
    {
      std::pair<unsigned int, unsigned int>
      lex_to_pair(const unsigned int rank,
                  const unsigned int size1,
                  const unsigned int size2)
      {
        AssertThrow(rank < size1 * size2, dealii::ExcMessage("Invalid rank."));
        return {rank % size1, rank / size1};
      }

      MPI_Comm
      create_rectangular_comm(const MPI_Comm &   comm,
                              const unsigned int size_0,
                              const unsigned int size_1)
      {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        AssertThrow((size_0 * size_1) <= static_cast<unsigned int>(size),
                    dealii::ExcMessage("Not enough ranks."));

        MPI_Comm sub_comm;
        MPI_Comm_split(comm,
                       (static_cast<unsigned int>(rank) < (size_0 * size_1)),
                       rank,
                       &sub_comm);

        if (static_cast<unsigned int>(rank) < (size_0 * size_1))
          return sub_comm;
        else
          {
            MPI_Comm_free(&sub_comm);
            return MPI_COMM_NULL;
          }
      }

      MPI_Comm
      create_row_comm(const MPI_Comm &   comm,
                      const unsigned int size1,
                      const unsigned int size2)
      {
        int size, rank;
        MPI_Comm_size(comm, &size);
        AssertThrow(static_cast<unsigned int>(size) == size1 * size2,
                    dealii::ExcMessage("Invalid communicator size."));

        MPI_Comm_rank(comm, &rank);

        MPI_Comm row_comm;
        MPI_Comm_split(comm,
                       lex_to_pair(rank, size1, size2).second,
                       rank,
                       &row_comm);
        return row_comm;
      }

      MPI_Comm
      create_column_comm(const MPI_Comm &   comm,
                         const unsigned int size1,
                         const unsigned int size2)
      {
        int size, rank;
        MPI_Comm_size(comm, &size);
        AssertThrow(static_cast<unsigned int>(size) == size1 * size2,
                    dealii::ExcMessage("Invalid communicator size."));

        MPI_Comm_rank(comm, &rank);

        MPI_Comm col_comm;
        MPI_Comm_split(comm,
                       lex_to_pair(rank, size1, size2).first,
                       rank,
                       &col_comm);
        return col_comm;
      }
    } // namespace MPI
  }   // namespace Utilities
} // namespace dealii



struct Parameters
{
  unsigned int n_refinements_0 = 6;
  unsigned int n_refinements_1 = 6;

  unsigned int fe_degree_0 = 1;
  unsigned int fe_degree_1 = 1;

  bool use_tensor_contraction = true;
};



template <int dim_0, int dim_1>
class Application
{
public:
  static const int dim = dim_0 + dim_1;

  // consructor
  Application(const Parameters params,
              const MPI_Comm   comm,
              const MPI_Comm   comm_0,
              const MPI_Comm   comm_1)
    : params(params)
    , comm(comm)
    , comm_0(comm_0)
    , comm_1(comm_1)
    , tria_0(comm_0, Triangulation<dim_0>::none, true)
    , dof_handler_0(tria_0)
    , fe_0(params.fe_degree_0)
    , quadrature_0(params.fe_degree_0 + 1)
    , tria_1(comm_1, Triangulation<dim_1>::none, true)
    , dof_handler_1(tria_1)
    , fe_1(params.fe_degree_1)
    , quadrature_1(params.fe_degree_1 + 1)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
  {}

  // main fuction
  void
  run()
  {
    create_grid();
    create_system();
    assemble_system();
    solve_system();
    print_results();
  }

private:
  // create hypercube with all boundary faces getting bid=0
  template <int dim_>
  void
  create_hyper_cube(Triangulation<dim_> &tria)
  {
    GridGenerator::hyper_cube(tria);

    const auto cell = tria.begin();

    for (const auto face : cell->face_iterators())
      face->set_boundary_id(0);
  }

  // create grid
  void
  create_grid()
  {
    create_hyper_cube(tria_0);
    tria_0.refine_global(params.n_refinements_0);

    create_hyper_cube(tria_1);
    tria_1.refine_global(params.n_refinements_1);

    const types::global_dof_index n_cells_0 = tria_0.n_global_active_cells();
    const types::global_dof_index n_cells_1 = tria_1.n_global_active_cells();
    const types::global_dof_index n_cells   = n_cells_0 * n_cells_1;

    pcout << "Created mesh with " << n_cells << " (" << n_cells_0 << "x"
          << n_cells_1 << ") cells." << std::endl;

    const auto print_partitioning = [&](const auto &tria) {
      unsigned int n_locally_owned_cells = 0;

      for (const auto &cell :
           tria.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
        {
          (void)cell;
          n_locally_owned_cells++;
        }

      const auto all_n_locally_owned_cells =
        Utilities::MPI::gather(tria.get_communicator(), n_locally_owned_cells);

      pcout << "  - with paritioning: ";
      for (unsigned int i = 0; i < all_n_locally_owned_cells.size(); ++i)
        {
          if (i != 0)
            pcout << ", ";
          pcout << all_n_locally_owned_cells[i];
        }
      pcout << std::endl;
    };

    print_partitioning(tria_0);
    print_partitioning(tria_1);

    pcout << std::endl;
  }

  // setup system
  void
  create_system()
  {
    // 1) distribute DoFs
    dof_handler_0.distribute_dofs(fe_0);
    dof_handler_1.distribute_dofs(fe_1);

    pcout << "Number of DoFs: "
          << (dof_handler_0.n_dofs() * dof_handler_1.n_dofs()) << " ("
          << dof_handler_0.n_dofs() << "x" << dof_handler_1.n_dofs() << ")."
          << std::endl
          << std::endl;

    constraints_0.reinit(
      DoFTools::extract_locally_relevant_dofs(dof_handler_0));
    DoFTools::make_zero_boundary_constraints(dof_handler_0, 0, constraints_0);
    constraints_0.close();

    constraints_1.reinit(
      DoFTools::extract_locally_relevant_dofs(dof_handler_1));
    DoFTools::make_zero_boundary_constraints(dof_handler_1, 0, constraints_1);
    constraints_1.close();

    // 2) "renumber" DoFs globally so that locally-owned DoFs are contiguous
    const auto is_local_0 = dof_handler_0.locally_owned_dofs();
    const auto is_active_0 =
      DoFTools::extract_locally_active_dofs(dof_handler_0);

    const auto is_local_1 = dof_handler_1.locally_owned_dofs();
    const auto is_active_1 =
      DoFTools::extract_locally_active_dofs(dof_handler_1);

    const auto is_local  = is_local_1.tensor_product(is_local_0);
    const auto is_active = is_active_1.tensor_product(is_active_0);

    const auto new_local_dof_order = generate_order_along_cell_order(is_local);

    const auto [is_local_new, is_active_new, permutation_active] =
      perform_renumbering(is_local, is_active, new_local_dof_order, comm);

    this->is_active          = is_active;
    this->permutation_active = permutation_active;
    this->partitioner =
      std::make_shared<const Utilities::MPI::Partitioner>(is_local_new,
                                                          is_active_new,
                                                          comm);

    // 3) create sparsity pattern
    sparsity_pattern.reinit(is_local_new, comm);

    std::vector<types::global_dof_index> dof_indices_0(fe_0.n_dofs_per_cell());
    std::vector<types::global_dof_index> dof_indices_1(fe_1.n_dofs_per_cell());

    for (const auto &cell_1 : dof_handler_1.active_cell_iterators() |
                                IteratorFilters::LocallyOwnedCell())
      {
        cell_1->get_dof_indices(dof_indices_1);

        for (const auto &cell_0 : dof_handler_0.active_cell_iterators() |
                                    IteratorFilters::LocallyOwnedCell())
          {
            cell_0->get_dof_indices(dof_indices_0);

            std::vector<types::global_dof_index> dofs;

            for (const auto i_1 : dof_indices_1)
              for (const auto i_0 : dof_indices_0)
                dofs.push_back(combine_index(i_0, i_1));

            std::sort(dofs.begin(), dofs.end());

            for (const auto i : dofs)
              sparsity_pattern.add_entries(i, dofs.begin(), dofs.end(), true);
          }
      }

    sparsity_pattern.compress();

    // 4) allocate memory for matrix and vectors
    sparse_matrix.reinit(sparsity_pattern);
    solution_vector.reinit(partitioner);
    rhs_vector.reinit(partitioner);
  }

  // assemble matrix and right-hand-side vector
  void
  assemble_system()
  {
    std::vector<types::global_dof_index> dof_indices_0(fe_0.n_dofs_per_cell());
    std::vector<types::global_dof_index> dof_indices_1(fe_1.n_dofs_per_cell());

    const unsigned int n_dofs_0 = fe_0.n_dofs_per_cell();
    const unsigned int n_dofs_1 = fe_1.n_dofs_per_cell();
    const unsigned int n_dofs = fe_0.n_dofs_per_cell() * fe_1.n_dofs_per_cell();

    FullMatrix<double> local_matrix(n_dofs, n_dofs);
    Vector<double>     local_vector(n_dofs);

    FullMatrix<double> M_0(n_dofs_0, n_dofs_0);
    FullMatrix<double> K_0(n_dofs_0, n_dofs_0);
    FullMatrix<double> M_1(n_dofs_1, n_dofs_1);
    FullMatrix<double> K_1(n_dofs_1, n_dofs_1);

    const auto update_flags =
      update_values | update_gradients | update_JxW_values;
    FEValues<dim_0> phi_0(mapping_0, fe_0, quadrature_0, update_flags);
    FEValues<dim_1> phi_1(mapping_1, fe_1, quadrature_1, update_flags);

    for (const auto &cell_1 : dof_handler_1.active_cell_iterators() |
                                IteratorFilters::LocallyOwnedCell())
      {
        cell_1->get_dof_indices(dof_indices_1);
        phi_1.reinit(cell_1);

        for (const auto &cell_0 : dof_handler_0.active_cell_iterators() |
                                    IteratorFilters::LocallyOwnedCell())
          {
            cell_0->get_dof_indices(dof_indices_0);
            phi_0.reinit(cell_0);

            local_matrix = 0.0;
            local_vector = 0.0;

            std::vector<double> f(n_dofs, 1.0);

            if (params.use_tensor_contraction == false)
              {
                for (const unsigned int q_1 : phi_1.quadrature_point_indices())
                  for (const unsigned int q_0 :
                       phi_0.quadrature_point_indices())
                    for (const unsigned int i_1 : phi_1.dof_indices())
                      for (const unsigned int i_0 : phi_0.dof_indices())
                        {
                          for (const unsigned int j_1 : phi_1.dof_indices())
                            for (const unsigned int j_0 : phi_0.dof_indices())
                              local_matrix(i_0 + i_1 * fe_0.n_dofs_per_cell(),
                                           j_0 +
                                             j_1 * fe_0.n_dofs_per_cell()) +=
                                ((phi_1.shape_value(i_1, q_1) *
                                  phi_1.shape_value(j_1, q_1) *
                                  phi_0.shape_grad(i_0, q_0) *
                                  phi_0.shape_grad(j_0, q_0)) +
                                 (phi_1.shape_grad(i_1, q_1) *
                                  phi_1.shape_grad(j_1, q_1) *
                                  phi_0.shape_value(i_0, q_0) *
                                  phi_0.shape_value(j_0, q_0))) *
                                (phi_0.JxW(q_0) * phi_1.JxW(q_1));

                          local_vector(i_0 + i_1 * fe_0.n_dofs_per_cell()) +=
                            f[i_0 + i_1 * fe_0.n_dofs_per_cell()] *
                            (phi_0.shape_value(i_0, q_0) *
                             phi_1.shape_value(i_1, q_1) * phi_0.JxW(q_0) *
                             phi_1.JxW(q_1));
                        }
              }
            else
              {
                M_0 = 0.0;
                K_0 = 0.0;
                M_1 = 0.0;
                K_1 = 0.0;

                for (const unsigned int q : phi_0.quadrature_point_indices())
                  for (const unsigned int i : phi_0.dof_indices())
                    for (const unsigned int j : phi_0.dof_indices())
                      {
                        M_0(i, j) += phi_0.shape_value(i, q) *
                                     phi_0.shape_value(j, q) * phi_0.JxW(q);

                        K_0(i, j) += phi_0.shape_grad(i, q) *
                                     phi_0.shape_grad(j, q) * phi_0.JxW(q);
                      }

                for (const unsigned int q : phi_1.quadrature_point_indices())
                  for (const unsigned int i : phi_1.dof_indices())
                    for (const unsigned int j : phi_1.dof_indices())
                      {
                        M_1(i, j) += phi_1.shape_value(i, q) *
                                     phi_1.shape_value(j, q) * phi_1.JxW(q);

                        K_1(i, j) += phi_1.shape_grad(i, q) *
                                     phi_1.shape_grad(j, q) * phi_1.JxW(q);
                      }

                for (const unsigned int i_1 : phi_1.dof_indices())
                  for (const unsigned int j_1 : phi_1.dof_indices())
                    for (const unsigned int i_0 : phi_0.dof_indices())
                      for (const unsigned int j_0 : phi_0.dof_indices())
                        {
                          local_matrix(i_0 + i_1 * fe_0.n_dofs_per_cell(),
                                       j_0 + j_1 * fe_0.n_dofs_per_cell()) =
                            M_1(i_1, j_1) * K_0(i_0, j_0) +
                            K_1(i_1, j_1) * M_0(i_0, j_0);

                          local_vector(i_0 + i_1 * fe_0.n_dofs_per_cell()) +=
                            f[j_0 + j_1 * fe_0.n_dofs_per_cell()] *
                            M_1(i_1, j_1) * M_0(i_0, j_0);
                        }
              }

            distribute_local_to_global(local_matrix,
                                       dof_indices_0,
                                       dof_indices_1,
                                       sparse_matrix);
            distribute_local_to_global(local_vector,
                                       dof_indices_0,
                                       dof_indices_1,
                                       rhs_vector);
          }
      }


    sparse_matrix.compress(VectorOperation::add);
    rhs_vector.compress(VectorOperation::add);
  }

  // solver the system with CG + AMG
  void
  solve_system()
  {
    TrilinosWrappers::PreconditionAMG amg;
    amg.initialize(sparse_matrix);

    ReductionControl solver_control(100, 1e-20, 1e-6);

    SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);
    solver.solve(sparse_matrix, solution_vector, rhs_vector, amg);

    pcout << "Solved in " << solver_control.last_step() << " iterations."
          << std::endl
          << std::endl;
  }

  // output results to Paraview; we output the result at the support points
  // directly as particles
  void
  print_results()
  {
    if constexpr (dim <= 3)
      {
        solution_vector.update_ghost_values();

        Triangulation<dim> tria;
        GridGenerator::hyper_cube(tria);

        MappingQ1<dim> mapping;

        Particles::ParticleHandler<dim> particle_handler(tria, mapping, 1);

        std::vector<Point<dim>>          points;
        std::vector<std::vector<double>> properties;


        std::vector<types::global_dof_index> dof_indices_0(
          fe_0.n_dofs_per_cell());
        std::vector<types::global_dof_index> dof_indices_1(
          fe_1.n_dofs_per_cell());

        FEValues<dim_0> phi_0(mapping_0,
                              fe_0,
                              fe_0.get_unit_support_points(),
                              update_quadrature_points);
        FEValues<dim_1> phi_1(mapping_1,
                              fe_1,
                              fe_1.get_unit_support_points(),
                              update_quadrature_points);


        for (const auto &cell_1 : dof_handler_1.active_cell_iterators() |
                                    IteratorFilters::LocallyOwnedCell())
          {
            cell_1->get_dof_indices(dof_indices_1);
            phi_1.reinit(cell_1);

            for (const auto &cell_0 : dof_handler_0.active_cell_iterators() |
                                        IteratorFilters::LocallyOwnedCell())
              {
                cell_0->get_dof_indices(dof_indices_0);
                phi_0.reinit(cell_0);

                for (const auto q_1 : phi_1.quadrature_point_indices())
                  for (const auto q_0 : phi_0.quadrature_point_indices())
                    {
                      const auto global_index =
                        combine_index(dof_indices_0[q_0], dof_indices_1[q_1]);

                      if (partitioner->in_local_range(global_index) == false)
                        continue;

                      Point<dim> point;

                      for (unsigned int d = 0; d < dim_0; ++d)
                        point[d] = phi_0.quadrature_point(q_0)[d];
                      for (unsigned int d = 0; d < dim_1; ++d)
                        point[d + dim_0] = phi_1.quadrature_point(q_1)[d];

                      points.push_back(point);

                      std::vector<double> property;
                      property.push_back(solution_vector[global_index]);
                      properties.push_back(property);
                    }
              }
          }

        // create bounding boxes
        std::vector<BoundingBox<dim>> local_boxes;
        for (const auto &cell :
             tria.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
          local_boxes.push_back(mapping.get_bounding_box(cell));
        const auto local_tree        = pack_rtree(local_boxes);
        const auto local_reduced_box = extract_rtree_level(local_tree, 1);
        const auto global_bounding_boxes =
          Utilities::MPI::all_gather(tria.get_communicator(),
                                     local_reduced_box);

        particle_handler.insert_global_particles(points,
                                                 global_bounding_boxes,
                                                 properties);

        Particles::DataOut<dim> data_out;
        data_out.build_patches(
          particle_handler,
          {"solution"},
          {DataComponentInterpretation::component_is_scalar});
        data_out.write_vtu_in_parallel("results.vtu", comm);

        solution_vector.zero_out_ghost_values();
      }
  }

  // renumber indices in @p is_local in the order they appear along the cell order
  std::vector<unsigned int>
  generate_order_along_cell_order(const IndexSet &is_local)
  {
    std::vector<unsigned int> new_order(is_local.n_elements(),
                                        numbers::invalid_unsigned_int);

    std::vector<types::global_dof_index> dof_indices_0(fe_0.n_dofs_per_cell());
    std::vector<types::global_dof_index> dof_indices_1(fe_1.n_dofs_per_cell());

    unsigned int counter = 0;

    for (const auto &cell_1 : dof_handler_1.active_cell_iterators() |
                                IteratorFilters::LocallyOwnedCell())
      {
        cell_1->get_dof_indices(dof_indices_1);

        for (const auto &cell_0 : dof_handler_0.active_cell_iterators() |
                                    IteratorFilters::LocallyOwnedCell())
          {
            cell_0->get_dof_indices(dof_indices_0);

            std::vector<types::global_dof_index> dofs;

            for (const auto i_1 : dof_indices_1)
              for (const auto i_0 : dof_indices_0)
                {
                  const auto index = i_0 + i_1 * dof_handler_0.n_dofs();

                  if (is_local.is_element(index))
                    {
                      const auto i = is_local.index_within_set(index);

                      if (new_order[i] == numbers::invalid_unsigned_int)
                        new_order[i] = counter++;
                    }
                }
          }
      }

    return new_order;
  }

  // create a new global numbering so that locally-owned indices are contiguous
  std::tuple<IndexSet, IndexSet, std::vector<types::global_dof_index>>
  perform_renumbering(const IndexSet &                 is_local,
                      const IndexSet &                 is_ghost,
                      const std::vector<unsigned int> &new_order,
                      const MPI_Comm &                 comm)
  {
    types::global_dof_index n_locally_owned_dofs = is_local.n_elements();
    types::global_dof_index offset               = 0;

    int ierr =
      MPI_Exscan(&n_locally_owned_dofs,
                 &offset,
                 1,
                 Utilities::MPI::mpi_type_id_for_type<types::global_dof_index>,
                 MPI_SUM,
                 comm);
    AssertThrowMPI(ierr);

    std::vector<unsigned int> remote_owner(is_ghost.n_elements());

    Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmsPayload
      process(is_local, is_ghost, comm, remote_owner, true);

    Utilities::MPI::ConsensusAlgorithms::Selector<
      std::vector<
        std::pair<types::global_cell_index, types::global_cell_index>>,
      std::vector<unsigned int>>
      consensus_algorithm;
    consensus_algorithm.run(process, comm);

    const auto targets_with_indexset = process.get_requesters();

    std::vector<unsigned int> targets;
    std::map<
      unsigned int,
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>
      data;

    for (const auto &target_with_indexset : targets_with_indexset)
      {
        const auto rank = target_with_indexset.first;

        targets.push_back(rank);

        std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
          load;
        load.reserve(target_with_indexset.second.n_elements());

        for (const auto i : target_with_indexset.second)
          {
            const auto i_local = is_local.index_within_set(i);
            load.emplace_back(
              i, offset + (new_order.empty() ? i_local : new_order[i_local]));
          }

        data[rank] = load;
      }

    std::vector<types::global_dof_index> new_dof_indices(
      is_ghost.n_elements(), numbers::invalid_dof_index);

    Utilities::MPI::ConsensusAlgorithms::selector<
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>(
      targets,
      [&data](const auto rank) { return data[rank]; },
      [&is_ghost, &new_dof_indices](const auto, const auto &answer) {
        for (const auto i : answer)
          new_dof_indices[is_ghost.index_within_set(i.first)] = i.second;
      },
      comm);


    IndexSet is_local_new(is_local.size());
    is_local_new.add_range(offset, offset + is_local.n_elements());

    const auto permutation_ghost = new_dof_indices;

    std::sort(new_dof_indices.begin(), new_dof_indices.end());

    IndexSet is_ghost_new(is_ghost.size());
    is_ghost_new.add_indices(new_dof_indices.begin(), new_dof_indices.end());

    return {is_local_new, is_ghost_new, permutation_ghost};
  }

  // combine two global indices
  types::global_dof_index
  combine_index(const types::global_dof_index i_0,
                const types::global_dof_index i_1) const
  {
    const unsigned int i =
      is_active.index_within_set(i_0 + i_1 * dof_handler_0.n_dofs());

    AssertIndexRange(i, permutation_active.size());

    return permutation_active[i];
  }

  // similar to AffineConstraints::distribute_local_to_global() but taking two
  // sets of global indices -> for vectors
  void
  distribute_local_to_global(
    const Vector<double>                        local_vector,
    const std::vector<types::global_dof_index> &dof_indices_0,
    const std::vector<types::global_dof_index> &dof_indices_1,
    LinearAlgebra::distributed::Vector<double> &vector)
  {
    std::vector<types::global_dof_index> dof_indices(local_vector.size());
    Vector<double>                       local_vector_copy = local_vector;

    for (unsigned int i_1 = 0, k = 0; i_1 < dof_indices_1.size(); ++i_1)
      for (unsigned int i_0 = 0; i_0 < dof_indices_0.size(); ++i_0, ++k)
        {
          const auto index_0 = dof_indices_0[i_0];
          const auto index_1 = dof_indices_1[i_1];

          dof_indices[k] = combine_index(index_0, index_1);

          if (constraints_0.is_constrained(index_0) ||
              constraints_1.is_constrained(index_1))
            local_vector_copy[k] = 0.0;
        }

    AffineConstraints<double> dummy;
    dummy.distribute_local_to_global(local_vector_copy, dof_indices, vector);
  }

  // similar to AffineConstraints::distribute_local_to_global() but taking two
  // sets of global indices -> for matrices
  void
  distribute_local_to_global(
    const FullMatrix<double>                    local_matrix,
    const std::vector<types::global_dof_index> &dof_indices_0,
    const std::vector<types::global_dof_index> &dof_indices_1,
    TrilinosWrappers::SparseMatrix &            matrix)
  {
    std::vector<types::global_dof_index> dof_indices(dof_indices_0.size() *
                                                     dof_indices_1.size());
    FullMatrix<double>                   local_matrix_copy = local_matrix;

    for (unsigned int i_1 = 0, k = 0; i_1 < dof_indices_1.size(); ++i_1)
      for (unsigned int i_0 = 0; i_0 < dof_indices_0.size(); ++i_0, ++k)
        {
          const auto index_0 = dof_indices_0[i_0];
          const auto index_1 = dof_indices_1[i_1];

          dof_indices[k] = combine_index(index_0, index_1);

          if (constraints_0.is_constrained(index_0) ||
              constraints_1.is_constrained(index_1))
            {
              for (unsigned int i = 0; i < local_matrix_copy.m(); ++i)
                {
                  local_matrix_copy[i][k] = 0.0;
                  local_matrix_copy[k][i] = 0.0;
                }

              local_matrix_copy[k][k] = 1.0;
            }
        }

    AffineConstraints<double> dummy;
    dummy.distribute_local_to_global(local_matrix_copy, dof_indices, matrix);
  }

  // parameters
  Parameters params;

  // communicators
  const MPI_Comm comm;
  const MPI_Comm comm_0;
  const MPI_Comm comm_1;

  // system 0
  parallel::shared::Triangulation<dim_0> tria_0;
  DoFHandler<dim_0>                      dof_handler_0;
  FE_Q<dim_0>                            fe_0;
  MappingQ1<dim_0>                       mapping_0;
  QGauss<dim_0>                          quadrature_0;
  AffineConstraints<double>              constraints_0;

  // system 1
  parallel::shared::Triangulation<dim_1> tria_1;
  DoFHandler<dim_1>                      dof_handler_1;
  FE_Q<dim_1>                            fe_1;
  MappingQ1<dim_1>                       mapping_1;
  QGauss<dim_1>                          quadrature_1;
  AffineConstraints<double>              constraints_1;

  // parallel indices
  IndexSet                                           is_active;
  std::vector<types::global_dof_index>               permutation_active;
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
  TrilinosWrappers::SparsityPattern                  sparsity_pattern;

  // matrix and vectors
  TrilinosWrappers::SparseMatrix             sparse_matrix;
  LinearAlgebra::distributed::Vector<double> solution_vector;
  LinearAlgebra::distributed::Vector<double> rhs_vector;

  ConditionalOStream pcout;
};

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim_0 = 1;
  const unsigned int dim_1 = 1;

  const unsigned int n_mpi_processes =
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  AssertThrow(n_mpi_processes == 1 || argc == 3, ExcInternalError());

  const unsigned int size_0 = (n_mpi_processes == 1) ? 1 : std::atoi(argv[1]);
  const unsigned int size_1 = (n_mpi_processes == 1) ? 1 : std::atoi(argv[2]);

  AssertThrow(size_0 * size_1 <= n_mpi_processes, ExcInternalError());

  Parameters params;

  MPI_Comm comm_global =
    Utilities::MPI::create_rectangular_comm(MPI_COMM_WORLD, size_0, size_1);

  if (comm_global != MPI_COMM_NULL)
    {
      MPI_Comm row_comm =
        Utilities::MPI::create_row_comm(comm_global, size_0, size_1);
      MPI_Comm column_comm =
        Utilities::MPI::create_column_comm(comm_global, size_0, size_1);

      Application<dim_0, dim_1> app(params, comm_global, row_comm, column_comm);

      app.run();

      MPI_Comm_free(&column_comm);
      MPI_Comm_free(&row_comm);
      MPI_Comm_free(&comm_global);
    }
}