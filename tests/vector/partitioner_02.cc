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


// Update ghost values with the help of LinearAlgebra::SharedMPI::Partitioner
// for a dealii::parallel::fullydistributed::Triangulation.

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_generator.h>

#include <hyper.deal/lac/sm_vector.h>
#include <hyper.deal/matrix_free/vector_access_kernels.h>
#include <hyper.deal/matrix_free/vector_partitioner.h>

#include "../tests.h"

using namespace dealii;

template <int dim,
          int degree,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
void
test(const MPI_Comm &comm, const MPI_Comm &comm_sm, const bool do_buffering)
{
  // 1) create triangulation
  parallel::fullydistributed::Triangulation<dim> tria(comm);

  {
    dealii::Triangulation<dim> tria_serial(
      dealii::Triangulation<dim>::limit_level_difference_at_vertices);
    GridGenerator::subdivided_hyper_cube(tria_serial, 2);
    tria_serial.refine_global(dim == 2 ? 4 : 3);
    dealii::GridTools::partition_triangulation_zorder(
      dealii::Utilities::MPI::n_mpi_processes(comm), tria_serial, false);

    const auto construction_data = dealii::TriangulationDescription::Utilities::
      create_description_from_triangulation(tria_serial, comm);
    tria.create_triangulation(construction_data);
  }


  // 2) create dof_handler so that cells are enumerated globally uniquelly
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_DGQ<dim>(0));

  // 3) setup data structures for partitioner
  std::vector<types::global_dof_index> local_cells;
  std::vector<std::pair<types::global_dof_index, std::vector<unsigned int>>>
    local_ghost_faces;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_artificial())
        continue;

      std::vector<types::global_dof_index> id(1);
      cell->get_dof_indices(id);
      if (cell->is_locally_owned()) // local cell
        local_cells.emplace_back(id.front());
      else // ghost cell
        {
          std::vector<unsigned int> faces;

          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               face++)
            {
              if (cell->at_boundary(face) ||
                  !cell->neighbor(face)->is_locally_owned())
                continue;

              faces.push_back(face);
            }

          if (!faces.empty())
            local_ghost_faces.emplace_back(id.front(), faces);
        }
    }

  // 4) setup partitioner
  hyperdeal::internal::MatrixFreeFunctions::ShapeInfo<Number> si;
  si.template reinit<dim>(degree);

  hyperdeal::internal::MatrixFreeFunctions::Partitioner<Number> partitioner(si);
  partitioner.reinit(
    local_cells, local_ghost_faces, comm, comm_sm, do_buffering);

  // setup "vector"
  dealii::LinearAlgebra::SharedMPI::Vector<Number> vector;
  vector.reinit(comm,
                comm_sm,
                partitioner.local_size(),
                partitioner.ghost_size());

  Number *               data_this   = vector.begin();
  std::vector<Number *> &data_others = vector.other_values();

  // fill "vector" with values
  unsigned int local_size  = partitioner.local_size();
  unsigned int local_start = 0;

  MPI_Exscan(&local_size, &local_start, 1, MPI_INT, MPI_SUM, comm);

  for (unsigned int i = 0; i < partitioner.local_size(); i++)
    data_this[i] = i + local_start;

  partitioner.update_ghost_values(data_this, data_others);

  const auto &maps       = partitioner.get_maps();
  const auto &maps_ghost = partitioner.get_maps_ghost();

  using DA = hyperdeal::internal::MatrixFreeFunctions::
    VectorReaderWriterKernels<dim, degree, Number, VectorizedArrayType>;

  const int dofs_per_cell = Utilities::pow(degree + 1, dim);
  const int dofs_per_face = Utilities::pow(degree + 1, dim - 1);

  std::vector<Number> cell_values(Utilities::pow(degree + 1, dim));
  std::vector<Number> face_values(Utilities::pow(degree + 1, dim - 1));

  Table<2, unsigned int> face_to_cell_index_nodal;
  face_to_cell_index_nodal.reinit(GeometryInfo<dim>::faces_per_cell,
                                  dofs_per_face);
  for (auto f : GeometryInfo<dim>::face_indices())
    {
      const int points = degree + 1;

      // create indices for one surfaces
      const int d = f / 2; // direction
      const int s = f % 2; // left or right surface

      const int b1 = Utilities::pow(points, d + 1);
      const int b2 = (s == 0 ? 0 : (points - 1)) * Utilities::pow(points, d);

      const unsigned int r1 = Utilities::pow(points, dim - d - 1);
      const unsigned int r2 = Utilities::pow(points, d);

      // collapsed iteration
      for (unsigned int i = 0, k = 0; i < r1; i++)
        for (unsigned int j = 0; j < r2; j++)
          face_to_cell_index_nodal(f, k++) = i * b1 + b2 + j;
    }

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_artificial())
        continue;

      std::vector<types::global_dof_index> id(1);
      cell->get_dof_indices(id);
      if (cell->is_locally_owned()) // local cell
        {
          const auto pair = maps.at(id[0]);
          DA::gather(data_others[pair.first] + pair.second, cell_values.data());

          for (unsigned int i = 0; i < cell_values.size(); i++)
            AssertDimension(cell_values[i], id[0] * dofs_per_cell + i);

          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               face++)
            {
              DA::gather_face_internal(data_others[pair.first] + pair.second,
                                       face,
                                       face_values.data());

              for (unsigned int i = 0; i < face_values.size(); i++)
                AssertDimension(face_values[i],
                                id[0] * dofs_per_cell +
                                  face_to_cell_index_nodal[face][i]);
            }
        }
      else // ghost cell
        {
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               face++)
            {
              if (cell->at_boundary(face) ||
                  !cell->neighbor(face)->is_locally_owned())
                continue;

              const auto pair_shared = maps.find(id[0]);
              const auto pair_remote = maps_ghost.find({id[0], face});

              Assert(pair_shared != maps.end() ||
                       pair_remote != maps_ghost.end(),
                     StandardExceptions::ExcNotImplemented());

              if (pair_remote != maps_ghost.end())
                {
                  const auto pair_remote = maps_ghost.find({id[0], face});

                  DA::gather_face(data_others[pair_remote->second.first] +
                                    pair_remote->second.second,
                                  face /*dummy*/,
                                  face_values.data(),
                                  true);
                }
              else
                {
                  DA::gather_face_internal(
                    data_others[pair_shared->second.first] +
                      pair_shared->second.second,
                    face,
                    face_values.data());
                }

              for (unsigned int i = 0; i < face_values.size(); i++)
                AssertDimension(face_values[i],
                                id[0] * dofs_per_cell +
                                  face_to_cell_index_nodal[face][i]);
            }
        }
    }

  deallog << "Passed: ghost_value_update()!" << std::endl;

  if (!do_buffering)
    return;

  MPI_Barrier(comm);

  for (unsigned int i = 0; i < partitioner.local_size(); i++)
    data_this[i] = 0.0;

  partitioner.compress(data_this, data_others);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (!cell->is_artificial() && cell->is_locally_owned()) // local cell
      {
        std::vector<types::global_dof_index> id(1);
        cell->get_dof_indices(id);

        std::vector<Number> cell_values_temp(Utilities::pow(degree + 1, dim),
                                             0);

        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             face++)
          {
            if (cell->at_boundary(face))
              continue;

            if (!cell->neighbor_or_periodic_neighbor(face)->is_ghost())
              continue;

            for (unsigned int i = 0; i < face_values.size(); i++)
              cell_values_temp[face_to_cell_index_nodal[face][i]] +=
                id[0] * dofs_per_cell + face_to_cell_index_nodal[face][i];
          }

        const auto pair = maps.at(id[0]);
        DA::gather(data_others[pair.first] + pair.second, cell_values.data());

        for (unsigned int i = 0; i < cell_values.size(); i++)
          AssertDimension(cell_values[i], cell_values_temp[i]);
      }

  deallog << "Passed: compress()!" << std::endl;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  for (unsigned int do_buffering = 0; do_buffering <= 1; do_buffering++)
    {
      deallog.push("buffering=" + std::to_string(do_buffering));

      for (unsigned int i = 1; i <= Utilities::MPI::n_mpi_processes(comm); i++)
        {
          deallog.push("size=" + std::to_string(i));

          MPI_Comm comm_sm;

          const unsigned int rank = Utilities::MPI::this_mpi_process(comm);
          MPI_Comm_split(comm, rank / i, rank, &comm_sm);

          deallog.push("dim=2:deg=1");
          test<2, 1, double>(comm, comm_sm, do_buffering);
          deallog.pop();

          deallog.push("dim=2:deg=2");
          test<2, 2, double>(comm, comm_sm, do_buffering);
          deallog.pop();

          deallog.push("dim=3:deg=1");
          test<3, 1, double>(comm, comm_sm, do_buffering);
          deallog.pop();

          deallog.push("dim=3:deg=2");
          test<3, 2, double>(comm, comm_sm, do_buffering);
          deallog.pop();

          MPI_Comm_free(&comm_sm);
          deallog.pop();
        }
      deallog.pop();
    }
}