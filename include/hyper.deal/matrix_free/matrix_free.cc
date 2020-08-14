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

#include <deal.II/fe/fe_dgq.h>

#include <hyper.deal/matrix_free/matrix_free.h>

namespace hyperdeal
{
  namespace internal
  {
    struct CellInfo
    {
      /**
       * Constructor.
       */
      CellInfo()
        : gid(0)
        , rank(0)
      {}

      /**
       * Constructor.
       */
      CellInfo(dealii::types::global_dof_index gid, unsigned int rank)
        : gid(gid)
        , rank(rank)
      {}

      /**
       * Global id of cell.
       */
      dealii::types::global_dof_index gid;

      /**
       * Rank of cell.
       */
      unsigned int rank;
    };

    struct FaceInfo
    {
      FaceInfo()
        : gid(0)
        , rank(0)
        , no(0)
      {}

      FaceInfo(unsigned int gid, unsigned int rank, unsigned char no)
        : gid(gid)
        , rank(rank)
        , no(no)
      {}
      unsigned int  gid;
      unsigned int  rank;
      unsigned char no;
    };

    struct GlobalCellInfo
    {
      unsigned int max_batch_size;
      unsigned int n_cell_batches;

      std::vector<CellInfo>     cells_interior;
      std::vector<CellInfo>     cells_exterior;
      std::vector<CellInfo>     cells;
      std::vector<CellInfo>     cells_ecl;
      std::vector<unsigned int> cells_lid;

      std::vector<unsigned char> cells_fill;
      std::vector<unsigned char> faces_fill;

      std::vector<unsigned int> interior_face_no;
      std::vector<unsigned int> exterior_face_no;

      unsigned int
      compute_n_cell_batches() const
      {
        return n_cell_batches;
      }

      unsigned int
      compute_n_cell_and_ghost_batches() const
      {
        return this->cells_fill.size();
      }

      unsigned int
      compute_n_cells() const
      {
        if (max_batch_size == 1)
          return n_cell_batches;

        unsigned int n_cells = 0;
        for (unsigned int i = 0; i < this->n_cell_batches; i++)
          for (unsigned int v = 0; v < this->cells_fill[i]; v++)
            n_cells++;
        return n_cells;
      }

      unsigned int
      compute_n_cells_and_ghost() const
      {
        if (max_batch_size == 1)
          return cells.size();

        unsigned int n_cells = 0;
        for (unsigned int i = 0; i < this->cells_fill.size(); i++)
          for (unsigned int v = 0; v < this->cells_fill[i]; v++)
            n_cells++;
        return n_cells;
      }
    };

    struct GlobalCellInfoProcessor
    {
      GlobalCellInfoProcessor(const GlobalCellInfo &info)
        : info(info)
      {}

      std::pair<unsigned int, unsigned int>
      get_local_range() const
      {
        // 1) determine local range
        unsigned int i_min = std::numeric_limits<unsigned int>::max();
        unsigned int i_max = std::numeric_limits<unsigned int>::min();

        for (unsigned int i = 0; i < info.n_cell_batches; i++)
          for (unsigned int v = 0; v < info.cells_fill[i]; v++)
            {
              unsigned int gid = info.cells[i * info.max_batch_size + v].gid;
              i_min            = std::min(i_min, gid);
              i_max            = std::max(i_max, gid);
            }

        return {i_min, i_max};
      }

      std::vector<FaceInfo>
      get_ghost_faces(const std::pair<unsigned int, unsigned int> local_range,
                      const int                                   dim,
                      const bool ecl = false) const
      {
        const auto            i_min = local_range.first;
        const auto            i_max = local_range.second;
        std::vector<FaceInfo> ghosts_faces;

        if (ecl)
          {
            for (unsigned int i = 0; i < info.n_cell_batches; i++)
              for (int d = 0; d < 2 * dim; d++)
                for (unsigned int v = 0; v < info.max_batch_size; v++)
                  {
                    Assert(i * info.max_batch_size * 2 * dim +
                               d * info.max_batch_size + v <
                             info.cells_ecl.size(),
                           dealii::ExcMessage("Out of range!"));
                    const auto cell_info =
                      info.cells_ecl[i * info.max_batch_size * 2 * dim +
                                     d * info.max_batch_size + v];
                    unsigned int gid = cell_info.gid;
                    if (gid == dealii::numbers::invalid_unsigned_int)
                      continue;
                    if (gid < i_min || i_max < gid)
                      ghosts_faces.emplace_back(
                        gid,
                        cell_info.rank,
                        d ^ 1); // TODO: only for structural mesh
                  }
          }
        else
          {
            for (unsigned int i = 0; i < info.exterior_face_no.size(); i++)
              for (unsigned int v = 0; v < info.faces_fill[i]; v++)
                {
                  const auto cell_info =
                    info.cells_interior[i * info.max_batch_size + v];
                  unsigned int gid = cell_info.gid;
                  if (gid < i_min || i_max < gid)
                    ghosts_faces.emplace_back(gid,
                                              cell_info.rank,
                                              info.interior_face_no[i]);
                }

            for (unsigned int i = 0; i < info.exterior_face_no.size(); i++)
              for (unsigned int v = 0; v < info.faces_fill[i]; v++)
                {
                  const auto cell_info =
                    info.cells_exterior[i * info.max_batch_size + v];
                  unsigned int gid = cell_info.gid;
                  if (gid < i_min || i_max < gid)
                    ghosts_faces.emplace_back(gid,
                                              cell_info.rank,
                                              info.exterior_face_no[i]);
                }
          }

        return ghosts_faces;
      }

      std::vector<FaceInfo>
      get_ghost_faces(const int dim, const bool ecl = false) const
      {
        return get_ghost_faces(get_local_range(), dim, ecl);
      }

      const GlobalCellInfo &info;
    };


    template <int dim, typename Number, typename VectorizedArrayType>
    void
    collect_global_cell_info(
      const dealii::MatrixFree<dim, Number, VectorizedArrayType> &data,
      GlobalCellInfo &                                            info)
    {
      const dealii::Triangulation<dim> &triangulation =
        data.get_dof_handler().get_triangulation();

      dealii::FE_DGQ<dim> fe_1(0);
      // ... distribute degrees of freedoms
      dealii::DoFHandler<dim> dof_handler_1(triangulation);
      dof_handler_1.distribute_dofs(fe_1);



      const auto cell_to_gid =
        [&](const typename dealii::DoFHandler<dim, dim>::cell_iterator &cell) {
          dealii::DoFAccessor<dim, dim, dim, true> a(&triangulation,
                                                     cell->level(),
                                                     cell->index(),
                                                     &dof_handler_1);

          std::vector<dealii::types::global_dof_index> indices(1);
          a.get_dof_indices(indices);
          return CellInfo(indices[0], a.subdomain_id());
        };

      const unsigned int v_len = VectorizedArrayType::size();

      info.max_batch_size = v_len;
      info.n_cell_batches = data.n_cell_batches();

      // cells
      info.cells.resize(v_len *
                        (data.n_cell_batches() + data.n_ghost_cell_batches()));
      info.cells_ecl.resize(dealii::GeometryInfo<dim>::faces_per_cell * v_len *
                            data.n_cell_batches());
      info.cells_lid.resize(
        v_len * (data.n_cell_batches() + data.n_ghost_cell_batches()));
      info.cells_interior.resize(
        v_len * (data.n_inner_face_batches() + data.n_boundary_face_batches()));
      info.cells_exterior.resize(v_len * data.n_inner_face_batches());


      // fill
      info.cells_fill.resize(data.n_cell_batches() +
                             data.n_ghost_cell_batches());
      info.faces_fill.resize(data.n_inner_face_batches() +
                             data.n_boundary_face_batches());

      // face_no
      info.interior_face_no.resize(data.n_inner_face_batches() +
                                   data.n_boundary_face_batches());
      info.exterior_face_no.resize(data.n_inner_face_batches());

      // local cells
      for (unsigned int cell = 0;
           cell < data.n_cell_batches() + data.n_ghost_cell_batches();
           cell++)
        {
          info.cells_fill[cell] = data.n_components_filled(cell);

          unsigned int v = 0;
          for (; v < data.n_components_filled(cell); v++)
            {
              auto c_it = data.get_cell_iterator(cell, v);

              info.cells[cell * v_len + v] = cell_to_gid(c_it);
              info.cells_lid[cell * v_len + v] =
                data.get_dof_info(/*TODO*/)
                  .dof_indices_contiguous[2][cell * v_len + v] /
                data.get_dofs_per_cell();

              if (cell < data.n_cell_batches())
                for (unsigned int d = 0;
                     d < dealii::GeometryInfo<dim>::faces_per_cell;
                     d++)
                  {
                    AssertThrow(c_it->has_periodic_neighbor(d) ||
                                  !c_it->at_boundary(d),
                                dealii::ExcMessage(
                                  "Boundaries are not supported yet.")) info
                      .cells_ecl[cell * v_len *
                                   dealii::GeometryInfo<dim>::faces_per_cell +
                                 v_len * d + v] =
                      cell_to_gid(c_it->neighbor_or_periodic_neighbor(d));
                  }
            }
          for (; v < v_len; v++)
            {
              if (cell < data.n_cell_batches())
                for (unsigned int d = 0;
                     d < dealii::GeometryInfo<dim>::faces_per_cell;
                     d++)
                  {
                    info.cells_ecl[cell * v_len *
                                     dealii::GeometryInfo<dim>::faces_per_cell +
                                   v_len * d + v] = {
                      dealii::numbers::invalid_unsigned_int,
                      dealii::numbers::invalid_unsigned_int};
                  }
            }
        }

      // interior faces
      for (unsigned int face = 0;
           face < data.n_inner_face_batches() + data.n_boundary_face_batches();
           face++)
        {
          info.faces_fill[face] = data.n_active_entries_per_face_batch(face);
          info.interior_face_no[face] =
            data.get_face_info(face).interior_face_no;
          for (unsigned int v = 0;
               v < data.n_active_entries_per_face_batch(face);
               v++)
            {
              const auto cell = data.get_face_info(face).cells_interior[v];
              info.cells_interior[face * v_len + v] =
                cell_to_gid(data.get_cell_iterator(cell / v_len, cell % v_len));
            }
        }

      // exterior faces
      for (unsigned int face = 0; face < data.n_inner_face_batches(); face++)
        {
          info.exterior_face_no[face] =
            data.get_face_info(face).exterior_face_no;
          for (unsigned int v = 0;
               v < data.n_active_entries_per_face_batch(face);
               v++)
            {
              const auto cell = data.get_face_info(face).cells_exterior[v];
              info.cells_exterior[face * v_len + v] =
                cell_to_gid(data.get_cell_iterator(cell / v_len, cell % v_len));
            }
        }
    }


    class Translator2
    {
    public:
      Translator2(const GlobalCellInfo &info_x,
                  const GlobalCellInfo &info_v,
                  const MPI_Comm        comm_x,
                  const MPI_Comm        comm_v)
      {
        n1.resize(dealii::Utilities::MPI::n_mpi_processes(comm_x) + 1);
        n2.resize(dealii::Utilities::MPI::n_mpi_processes(comm_v) + 1);

        {
          std::vector<unsigned int> n1_temp(n1.size() - 1);
          unsigned int              n1_local = info_x.compute_n_cells();
          MPI_Allgather(
            &n1_local, 1, MPI_UNSIGNED, &n1_temp[0], 1, MPI_UNSIGNED, comm_x);
          for (unsigned int i = 0; i < n1.size() - 1; i++)
            n1[i + 1] = n1[i] + n1_temp[i];

          std::vector<unsigned int> n2_temp(n2.size() - 1);
          unsigned int              n2_local = info_v.compute_n_cells();
          MPI_Allgather(
            &n2_local, 1, MPI_UNSIGNED, &n2_temp[0], 1, MPI_UNSIGNED, comm_v);
          for (unsigned int i = 0; i < n2.size() - 1; i++)
            n2[i + 1] = n2[i] + n2_temp[i];
        }
      }

      CellInfo
      translate(unsigned int gid1,
                unsigned int rank1,
                unsigned int gid2,
                unsigned int rank2)
      {
        AssertThrow(rank1 < n1.size(),
                    dealii::ExcMessage("Size does not match."));
        AssertThrow(rank2 < n2.size(),
                    dealii::ExcMessage("Size does not match."));
        unsigned int lid1 = gid1 - n1[rank1];
        unsigned int lid2 = gid2 - n2[rank2];

        unsigned int lid = lid1 + lid2 * (n1[rank1 + 1] - n1[rank1]);

        return {lid + n1.back() * n2[rank2] +
                  n1[rank1] * (n2[rank2 + 1] - n2[rank2]),
                rank1 + ((unsigned int)n1.size() - 1) * rank2};
      }

    private:
      std::vector<unsigned int> n1;
      std::vector<unsigned int> n2;
    };

    void
    combine_global_cell_info(const MPI_Comm        comm_x,
                             const MPI_Comm        comm_v,
                             const GlobalCellInfo &info_x,
                             const GlobalCellInfo &info_v,
                             GlobalCellInfo &      info,
                             int                   dim_x,
                             int                   dim_v)
    {
      // batch size
      info.max_batch_size = info_x.max_batch_size;

      // number of cell batches
      info.n_cell_batches =
        info_x.compute_n_cell_batches() * info_v.compute_n_cells();

      Translator2 t(info_x, info_v, comm_x, comm_v);

      auto process = [&](unsigned int i_x, unsigned int i_v, unsigned int v_v) {
        unsigned int v_x = 0;
        for (; v_x < info_x.cells_fill[i_x]; v_x++)
          {
            const auto cell_x = info_x.cells[i_x * info_x.max_batch_size + v_x];
            const auto cell_y = info_v.cells[i_v * info_v.max_batch_size + v_v];
            info.cells.emplace_back(
              t.translate(cell_x.gid, cell_x.rank, cell_y.gid, cell_y.rank));
          }
        for (; v_x < info_x.max_batch_size; v_x++)
          info.cells.emplace_back(-1, -1);

        info.cells_fill.push_back(info_x.cells_fill[i_x]);
      };

      const unsigned int n_cell_batches_x = info_x.compute_n_cell_batches();
      const unsigned int n_cell_batches_v = info_v.compute_n_cell_batches();

      const unsigned int n_cell_and_ghost_batches_x =
        info_x.compute_n_cell_and_ghost_batches();
      const unsigned int n_cell_and_ghost_batches_v =
        info_v.compute_n_cell_and_ghost_batches();

      // cells: local x local
      for (unsigned int i_v = 0; i_v < n_cell_batches_v; i_v++)
        for (unsigned int v_v = 0; v_v < info_v.cells_fill[i_v]; v_v++)
          for (unsigned int i_x = 0; i_x < n_cell_batches_x; i_x++)
            process(i_x, i_v, v_v);

      // ecl neighbors
      for (unsigned int i_v = 0; i_v < n_cell_batches_v; i_v++)
        for (unsigned int v_v = 0; v_v < info_v.cells_fill[i_v]; v_v++)
          for (unsigned int i_x = 0; i_x < n_cell_batches_x; i_x++)
            {
              for (int d = 0; d < 2 * dim_x; d++)
                {
                  unsigned int v_x = 0;
                  for (; v_x < info_x.cells_fill[i_x]; v_x++)
                    {
                      const auto cell_x =
                        info_x
                          .cells_ecl[i_x * info_x.max_batch_size * 2 * dim_x +
                                     d * info_x.max_batch_size + v_x];
                      const auto cell_y =
                        info_v.cells[i_v * info_v.max_batch_size + v_v];
                      info.cells_ecl.emplace_back(t.translate(
                        cell_x.gid, cell_x.rank, cell_y.gid, cell_y.rank));
                    }
                  for (; v_x < info_x.max_batch_size; v_x++)
                    info.cells_ecl.emplace_back(-1, -1);
                }

              for (int d = 0; d < 2 * dim_v; d++)
                {
                  unsigned int v_x = 0;
                  for (; v_x < info_x.cells_fill[i_x]; v_x++)
                    {
                      const auto cell_x =
                        info_x.cells[i_x * info_x.max_batch_size + v_x];
                      const auto cell_y =
                        info_v
                          .cells_ecl[i_v * info_v.max_batch_size * 2 * dim_v +
                                     d * info_v.max_batch_size + v_v];
                      info.cells_ecl.emplace_back(t.translate(
                        cell_x.gid, cell_x.rank, cell_y.gid, cell_y.rank));
                    }
                  for (; v_x < info_x.max_batch_size; v_x++)
                    info.cells_ecl.emplace_back(-1, -1);
                }
            }

      // cells: ghost x local
      for (unsigned int i_v = 0; i_v < n_cell_batches_v; i_v++)
        for (unsigned int v_v = 0; v_v < info_v.cells_fill[i_v]; v_v++)
          for (unsigned int i_x = n_cell_batches_x;
               i_x < n_cell_and_ghost_batches_x;
               i_x++)
            process(i_x, i_v, v_v);

      // cells: local x ghost
      for (unsigned int i_v = n_cell_batches_v;
           i_v < n_cell_and_ghost_batches_v;
           i_v++)
        for (unsigned int v_v = 0; v_v < info_v.cells_fill[i_v]; v_v++)
          for (unsigned int i_x = 0; i_x < n_cell_batches_x; i_x++)
            process(i_x, i_v, v_v);

      // internal faces
      {
        const unsigned int n_face_batches_x = info_x.interior_face_no.size();
        const unsigned int n_face_batches_v = info_v.interior_face_no.size();
        // interior faces (face x cell):
        for (unsigned int i_v = 0; i_v < n_cell_batches_v; i_v++)
          for (unsigned int v_v = 0; v_v < info_v.cells_fill[i_v]; v_v++)
            for (unsigned int i_x = 0; i_x < n_face_batches_x; i_x++)
              {
                unsigned int v_x = 0;
                for (; v_x < info_x.faces_fill[i_x]; v_x++)
                  {
                    const auto cell_x =
                      info_x.cells_interior[i_x * info_x.max_batch_size + v_x];
                    const auto cell_y =
                      info_v.cells[i_v * info_v.max_batch_size + v_v];
                    info.cells_interior.emplace_back(t.translate(
                      cell_x.gid, cell_x.rank, cell_y.gid, cell_y.rank));
                  }
                for (; v_x < info_x.max_batch_size; v_x++)
                  info.cells_interior.emplace_back(-1, -1);

                info.faces_fill.push_back(info_x.faces_fill[i_x]);

                info.interior_face_no.push_back(info_x.interior_face_no[i_x]);
              }

        // interior faces (cell x face):
        for (unsigned int i_v = 0; i_v < n_face_batches_v; i_v++)
          for (unsigned int v_v = 0; v_v < info_v.faces_fill[i_v]; v_v++)
            for (unsigned int i_x = 0; i_x < n_cell_batches_x; i_x++)
              {
                unsigned int v_x = 0;
                for (; v_x < info_x.cells_fill[i_x]; v_x++)
                  {
                    const auto cell_x =
                      info_x.cells[i_x * info_x.max_batch_size + v_x];
                    const auto cell_y =
                      info_v.cells_interior[i_v * info_v.max_batch_size + v_v];
                    info.cells_interior.emplace_back(t.translate(
                      cell_x.gid, cell_x.rank, cell_y.gid, cell_y.rank));
                  }
                for (; v_x < info_x.max_batch_size; v_x++)
                  info.cells_interior.emplace_back(-1, -1);

                info.faces_fill.push_back(info_x.cells_fill[i_x]);

                info.interior_face_no.push_back(info_v.interior_face_no[i_v] +
                                                2 * dim_x);
              }
      }

      // external faces
      {
        const unsigned int n_face_batches_x = info_x.exterior_face_no.size();
        const unsigned int n_face_batches_v = info_v.exterior_face_no.size();
        // exterior faces (face x cell):
        for (unsigned int i_v = 0; i_v < n_cell_batches_v; i_v++)
          for (unsigned int v_v = 0; v_v < info_v.cells_fill[i_v]; v_v++)
            for (unsigned int i_x = 0; i_x < n_face_batches_x; i_x++)
              {
                unsigned int v_x = 0;
                for (; v_x < info_x.faces_fill[i_x]; v_x++)
                  {
                    const auto cell_x =
                      info_x.cells_exterior[i_x * info_x.max_batch_size + v_x];
                    const auto cell_y =
                      info_v.cells[i_v * info_v.max_batch_size + v_v];
                    info.cells_exterior.emplace_back(t.translate(
                      cell_x.gid, cell_x.rank, cell_y.gid, cell_y.rank));
                  }
                for (; v_x < info_x.max_batch_size; v_x++)
                  info.cells_exterior.emplace_back(-1, -1);

                info.exterior_face_no.push_back(info_x.exterior_face_no[i_x]);
              }

        // exterior faces (cell x face):
        for (unsigned int i_v = 0; i_v < n_face_batches_v; i_v++)
          for (unsigned int v_v = 0; v_v < info_v.faces_fill[i_v]; v_v++)
            for (unsigned int i_x = 0; i_x < n_cell_batches_x; i_x++)
              {
                unsigned int v_x = 0;
                for (; v_x < info_x.cells_fill[i_x]; v_x++)
                  {
                    const auto cell_x =
                      info_x.cells[i_x * info_x.max_batch_size + v_x];
                    const auto cell_y =
                      info_v.cells_exterior[i_v * info_v.max_batch_size + v_v];
                    info.cells_exterior.emplace_back(t.translate(
                      cell_x.gid, cell_x.rank, cell_y.gid, cell_y.rank));
                  }
                for (; v_x < info_x.max_batch_size; v_x++)
                  info.cells_exterior.emplace_back(-1, -1);

                info.exterior_face_no.push_back(info_v.exterior_face_no[i_v] +
                                                2 * dim_x);
              }
      }
    }
  } // namespace internal


  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::MatrixFree(
    const MPI_Comm &comm,
    const MPI_Comm &comm_sm,
    const dealii::MatrixFree<dim_x, Number, VectorizedArrayTypeX>
      &matrix_free_x,
    const dealii::MatrixFree<dim_v, Number, VectorizedArrayTypeV>
      &matrix_free_v)
    : comm(comm)
    , comm_sm(comm_sm)
    , matrix_free_x(matrix_free_x)
    , matrix_free_v(matrix_free_v)
  {}

  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  void
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::reinit(
    const AdditionalData &additional_data)
  {
    // store relevant settings
    this->do_buffering   = additional_data.do_buffering;
    this->do_ghost_faces = additional_data.do_ghost_faces;
    this->use_ecl        = additional_data.use_ecl;

    AssertThrow(this->do_ghost_faces,
                dealii::ExcMessage(
                  "At the moment only ghost faces are supported!"));

    AssertThrow(this->use_ecl || (/*!this->use_ecl &&*/ this->do_buffering),
                dealii::ExcMessage("FCL needs buffering!"));

    AssertThrow(additional_data.fe_degree !=
                  dealii::numbers::invalid_unsigned_int,
                dealii::ExcMessage("Degree has not been set!"));

    this->shape_info.template reinit<dim_x + dim_v>(additional_data.fe_degree);

    partitioner =
      std::make_shared<internal::MatrixFreeFunctions::Partitioner<Number>>(
        shape_info);

    reader_writer = std::make_shared<
      internal::MatrixFreeFunctions::VectorReaderWriter<Number>>(dof_info,
                                                                 face_info);

    const int dim = dim_x + dim_v;

    // collect (global) information of each macro cell in phase space
    const auto info = [&]() {
      internal::GlobalCellInfo info_x, info_v, info;

      // collect (global) information of each macro cell in x-space
      internal::collect_global_cell_info(matrix_free_x, info_x);

      // collect (global) information of each macro cell in v-space
      internal::collect_global_cell_info(matrix_free_v, info_v);

      // create tensor product
      internal::combine_global_cell_info(
        matrix_free_x.get_vector_partitioner(/*TODO*/)->get_mpi_communicator(),
        matrix_free_v.get_vector_partitioner(/*TODO*/)->get_mpi_communicator(),
        info_x,
        info_v,
        info,
        dim_x,
        dim_v);

      return info;
    }();


    {
      // create a list of inner cells and ghost faces
      std::vector<dealii::types::global_dof_index> local_list;
      std::vector<
        std::pair<dealii::types::global_dof_index, std::vector<unsigned int>>>
        ghost_list;

      // 1) collect local cells
      {
        for (unsigned int i = 0; i < info.n_cell_batches; i++)
          for (unsigned int v = 0; v < info.cells_fill[i]; v++)
            local_list.emplace_back(
              info.cells[i * info.max_batch_size + v].gid);
        std::sort(local_list.begin(), local_list.end());
      }

      // 2) collect ghost faces and group them for each cell
      {
        const internal::GlobalCellInfoProcessor gcip(info);
        auto ghost_faces = gcip.get_ghost_faces(dim, this->use_ecl);

        std::sort(ghost_faces.begin(),
                  ghost_faces.end(),
                  [](const auto &a, const auto &b) {
                    if (a.gid < b.gid)
                      return true;

                    if (a.gid == b.gid && a.no < b.no)
                      return true;

                    return false;
                  });

        if (ghost_faces.size() > 0)
          {
            auto ptr = ghost_faces.begin();

            ghost_list.emplace_back(ptr->gid,
                                    std::vector<unsigned int>{ptr->no});
            ptr++;

            for (; ptr != ghost_faces.end(); ptr++)
              {
                if (ghost_list.back().first == ptr->gid)
                  ghost_list.back().second.emplace_back(ptr->no);
                else
                  ghost_list.emplace_back(ptr->gid,
                                          std::vector<unsigned int>{ptr->no});
              }
          }
      }

      // setup partitioner
      if (do_ghost_faces)
        {
          // ghost faces only
          this->partitioner->reinit(
            local_list, ghost_list, comm, comm_sm, do_buffering);
        }
      else
        {
          // ghost cells only
          AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
        }
    }


    // set up rest of dof_info and face_info (1)
    {
      auto &n_vectorization_lanes_filled =
        dof_info.n_vectorization_lanes_filled;
      auto &dof_indices_contiguous = dof_info.dof_indices_contiguous;
      auto &no_faces               = face_info.no_faces;

      // 3) collect gids according to vectorization
      {
        for (unsigned int i = 0; i < info.interior_face_no.size(); i++)
          for (unsigned int v = 0; v < info.max_batch_size; v++)
            dof_indices_contiguous[0].push_back(
              info.cells_interior[i * info.max_batch_size + v].gid);
        for (unsigned int i = 0; i < info.exterior_face_no.size(); i++)
          for (unsigned int v = 0; v < info.max_batch_size; v++)
            dof_indices_contiguous[1].push_back(
              info.cells_exterior[i * info.max_batch_size + v].gid);
        for (unsigned int i = 0; i < info.n_cell_batches; i++)
          for (unsigned int v = 0; v < info.max_batch_size; v++)
            dof_indices_contiguous[2].push_back(
              info.cells[i * info.max_batch_size + v].gid);
        for (unsigned int i = 0; i < info.n_cell_batches; i++)
          for (unsigned int d = 0;
               d < dealii::GeometryInfo<dim>::faces_per_cell;
               d++)
            for (unsigned int v = 0; v < info.max_batch_size; v++)
              dof_indices_contiguous[3].push_back(
                info
                  .cells_ecl[i * info.max_batch_size *
                               dealii::GeometryInfo<dim>::faces_per_cell +
                             d * info.max_batch_size + v]
                  .gid);
      }

      // 4) collect filled lanes
      {
        for (unsigned int i = 0; i < info.interior_face_no.size(); i++)
          n_vectorization_lanes_filled[0].push_back(info.faces_fill[i]);

        for (unsigned int i = 0; i < info.exterior_face_no.size(); i++)
          n_vectorization_lanes_filled[1].push_back(info.faces_fill[i]);

        for (unsigned int i = 0; i < info.n_cell_batches; i++)
          n_vectorization_lanes_filled[2].push_back(info.cells_fill[i]);

        if (this->use_ecl) // TODO: why the hack are ECL data structures set up?
          for (unsigned int i = 0; i < info.n_cell_batches; i++)
            for (unsigned int d = 0;
                 d < dealii::GeometryInfo<dim>::faces_per_cell;
                 d++)
              n_vectorization_lanes_filled[3].push_back(info.cells_fill[i]);
      }

      // 5) collect face orientations
      {
        no_faces[0] = info.interior_face_no;
        no_faces[1] = info.exterior_face_no;
      }
    }


    // set up rest of dof_info and face_info (2)
    {
      // given information
      const auto &vectorization_length = dof_info.n_vectorization_lanes_filled;
      const auto &dof_indices_contiguous = dof_info.dof_indices_contiguous;
      const auto &no_faces               = face_info.no_faces;

      const auto &maps       = partitioner->get_maps();
      const auto &maps_ghost = partitioner->get_maps_ghost();

      // to be computed
      auto &cell_ptrs = dof_info.dof_indices_contiguous_ptr;
      auto &face_type = face_info.face_type;
      auto &face_all  = face_info.face_all;

      for (unsigned int i = 0; i < 4; i++)
        cell_ptrs[i].resize(dof_indices_contiguous[i].size());

      for (unsigned int i = 0; i < 4; i++)
        if (i != 2)
          face_type[i].resize(dof_indices_contiguous[i].size());

      for (unsigned int i = 0; i < 4; i++)
        if (i != 2)
          face_all[i].resize(vectorization_length[i].size());

      static const int v_len = VectorizedArrayType::size();

      for (unsigned int i = 0; i < 4; i++)
        if (i != 2)
          for (unsigned int j = 0; j < vectorization_length[i].size(); j++)
            for (unsigned int k = 0; k < vectorization_length[i][j]; k++)
              {
                unsigned int l = j * v_len + k;
                Assert(l < dof_indices_contiguous[i].size(),
                       dealii::StandardExceptions::ExcMessage(
                         "Size of gid does not match."));
                unsigned int gid_this = dof_indices_contiguous[i][l];

                Assert(gid_this != dealii::numbers::invalid_unsigned_int,
                       dealii::StandardExceptions::ExcMessage(
                         "Boundaries are not supported yet."));

                auto ptr1 = maps_ghost.find(
                  {gid_this,
                   do_ghost_faces ?
                     (i == 3 ?
                        (j % (dim * 2)) ^ 1 /*TODO: only for structural mesh*/ :
                        no_faces[i][j]) :
                     dealii::numbers::invalid_unsigned_int});

                if (ptr1 != maps_ghost.end())
                  {
                    cell_ptrs[i][l] = {ptr1->second.first, ptr1->second.second};
                    face_type[i][l] = true;
                    continue;
                  }

                auto ptr2 = maps.find(gid_this);

                if (ptr2 != maps.end())
                  {
                    cell_ptrs[i][l] = {ptr2->second.first, ptr2->second.second};
                    face_type[i][l] = false;
                    continue;
                  }

                AssertThrow(false,
                            dealii::StandardExceptions::ExcMessage(
                              "Cell not found!"));
              }

      for (unsigned int i = 0; i < 4; i++)
        if (i != 2)
          for (unsigned int j = 0; j < vectorization_length[i].size(); j++)
            {
              bool temp = true;
              for (unsigned int k = 0; k < vectorization_length[i][j]; k++)
                temp &= face_type[i][j * v_len] && face_type[i][j * v_len + k];
              face_all[i][j] = temp;
            }

      for (unsigned int i = 2; i < 3; i++)
        for (unsigned int j = 0; j < vectorization_length[i].size(); j++)
          for (unsigned int k = 0; k < vectorization_length[i][j]; k++)
            {
              unsigned int l = j * v_len + k;

              unsigned int gid_this = dof_indices_contiguous[i][l];
              auto         ptr      = maps.find(gid_this);

              // TODO: why?
              if (ptr == maps.end())
                continue;

              cell_ptrs[i][l] = {ptr->second.first, ptr->second.second};
            }
    }
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  void
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::initialize_dof_vector(
    dealii::LinearAlgebra::SharedMPI::Vector<Number> &vec,
    const unsigned int                                dof_handler_index,
    const bool                                        do_ghosts,
    const bool                                        zero_out_values) const
  {
    AssertThrow(dof_handler_index == 0,
                dealii::ExcMessage(
                  "Only one dof_handler supported at the moment!"));

    AssertThrow((partitioner != nullptr) && (reader_writer != nullptr),
                dealii::ExcMessage("Partitioner has not been initialized!"));

    // setup vector
    vec.reinit(comm,
               comm_sm,
               partitioner->local_size(),
               do_ghosts ? partitioner->ghost_size() : 0);

    // zero out values
    if (zero_out_values)
      vec.zero_out(true);

    // perform test ghost value update
    if (zero_out_values && do_ghosts)
      {
        // working for ECL/FCL
        partitioner->update_ghost_values_start(vec.begin(), vec.other_values());
        partitioner->update_ghost_values_finish(vec.begin(),
                                                vec.other_values());

        // working only for FCL
        if (!use_ecl)
          {
            partitioner->compress_start(vec.begin(), vec.other_values());
            partitioner->compress_finish(vec.begin(), vec.other_values());
          }
      }
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  template <typename OutVector, typename InVector>
  void
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::cell_loop(
    const std::function<
      void(const MatrixFree &, OutVector &, const InVector &, const ID)>
      &             cell_operation,
    OutVector &     dst,
    const InVector &src) const
  {
    unsigned int v_len = VectorizedArrayTypeV::size();

    const unsigned int n_cell_batches_x = matrix_free_x.n_cell_batches();
    const unsigned int n_cell_batches_v = matrix_free_v.n_cell_batches();

    unsigned int i0 = 0;

    for (unsigned int j = 0; j < n_cell_batches_v; j++)
      for (unsigned int v = 0;
           v < matrix_free_v.n_active_entries_per_cell_batch(j);
           v++)
        for (unsigned int i = 0; i < n_cell_batches_x; i++)
          cell_operation(*this, dst, src, ID(i, j * v_len + v, i0++));
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  template <typename CLASS, typename OutVector, typename InVector>
  void
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::cell_loop(
    void (CLASS::*cell_operation)(const MatrixFree &,
                                  OutVector &,
                                  const InVector &,
                                  const ID),
    CLASS *         owning_class,
    OutVector &     dst,
    const InVector &src) const
  {
    this->template cell_loop<OutVector, InVector>(
      [&cell_operation, &owning_class](const MatrixFree &mf,
                                       OutVector &       dst,
                                       const InVector &  src,
                                       const ID          id) {
        (owning_class->*cell_operation)(mf, dst, src, id);
      },
      dst,
      src);
  }

  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  template <typename CLASS, typename OutVector, typename InVector>
  void
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::loop_cell_centric(
    void (CLASS::*cell_operation)(const MatrixFree &,
                                  OutVector &,
                                  const InVector &,
                                  const ID),
    CLASS *                 owning_class,
    OutVector &             dst,
    const InVector &        src,
    const DataAccessOnFaces src_vector_face_access,
    Timers *                timers) const
  {
    if (src_vector_face_access == DataAccessOnFaces::values)
      {
        ScopedTimerWrapper timer(timers, "update_ghost_values");

        InVector &src_ = const_cast<InVector &>(src);
        partitioner->update_ghost_values_start(src_.begin(),
                                               src_.other_values());
        partitioner->update_ghost_values_finish(src_.begin(),
                                                src_.other_values());
      }
    else
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());

    {
      ScopedTimerWrapper timer(timers, "zero_out_ghosts");
      dst.zero_out_ghosts();
    }

    {
      ScopedTimerWrapper timer(timers, "loop");

      const unsigned int v_len            = VectorizedArrayTypeV::size();
      const unsigned int n_cell_batches_x = matrix_free_x.n_cell_batches();
      const unsigned int n_cell_batches_v = matrix_free_v.n_cell_batches();

      for (unsigned int j = 0, i0 = 0; j < n_cell_batches_v; j++)
        for (unsigned int v = 0;
             v < matrix_free_v.n_active_entries_per_cell_batch(j);
             v++)
          for (unsigned int i = 0; i < n_cell_batches_x; i++, i0++)
            (owning_class->*cell_operation)(*this,
                                            dst,
                                            src,
                                            ID(i, j * v_len + v, i0));
    }

    if (!do_buffering)
      {
        ScopedTimerWrapper timer(timers, "barrier");
        partitioner->sync();
      }
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  template <typename CLASS, typename OutVector, typename InVector>
  void
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::loop(
    void (CLASS::*cell_operation)(const MatrixFree &,
                                  OutVector &,
                                  const InVector &,
                                  const ID),
    void (CLASS::*face_operation)(const MatrixFree &,
                                  OutVector &,
                                  const InVector &,
                                  const ID),
    void (CLASS::*boundary_operation)(const MatrixFree &,
                                      OutVector &,
                                      const InVector &,
                                      const ID),
    CLASS *                 owning_class,
    OutVector &             dst,
    const InVector &        src,
    const DataAccessOnFaces dst_vector_face_access,
    const DataAccessOnFaces src_vector_face_access,
    Timers *                timers) const
  {
    if (src_vector_face_access == DataAccessOnFaces::values)
      {
        ScopedTimerWrapper timer(timers, "update_ghost_values");
        InVector &         src_ = const_cast<InVector &>(src);
        partitioner->update_ghost_values_start(src_.begin(),
                                               src_.other_values());
        partitioner->update_ghost_values_finish(src_.begin(),
                                                src_.other_values());
      }
    else
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());

    {
      ScopedTimerWrapper timer(timers, "zero_out_ghosts");
      dst.zero_out_ghosts();
    }

    unsigned int v_len = VectorizedArrayTypeV::size();

    const unsigned int n_cell_batches_x = matrix_free_x.n_cell_batches();
    const unsigned int n_inner_face_batches_x =
      matrix_free_x.n_inner_face_batches();
    const unsigned int n_inner_boundary_batches_x =
      matrix_free_x.n_boundary_face_batches();
    const unsigned int n_inner_or_boundary_face_batches_x =
      n_inner_face_batches_x + n_inner_boundary_batches_x;

    const unsigned int n_cell_batches_v = matrix_free_v.n_cell_batches();
    const unsigned int n_inner_face_batches_v =
      matrix_free_v.n_inner_face_batches();
    const unsigned int n_inner_boundary_batches_v =
      matrix_free_v.n_boundary_face_batches();
    const unsigned int n_inner_or_boundary_face_batches_v =
      n_inner_face_batches_v + n_inner_boundary_batches_v;

    unsigned int i0 = 0;
    unsigned int i1 = 0;
    unsigned int i2 = 0;

    // clang-format off
  
    // loop over all cells
    {
      ScopedTimerWrapper timer(timers, "cell_loop");
      for(unsigned int j = 0; j < n_cell_batches_v; j++)
        for(unsigned int v = 0; v < matrix_free_v.n_active_entries_per_cell_batch(j); v++)
          for(unsigned int i = 0; i < n_cell_batches_x; i++)
            (owning_class->*cell_operation)(*this, dst, src, ID(i, j * v_len + v, i0++));
    }
  
    // loop over all inner faces ...
    {
      ScopedTimerWrapper timer(timers, "face_loop_x");
      for(unsigned int j = 0; j < n_cell_batches_v; j++)
        for(unsigned int v = 0; v < matrix_free_v.n_active_entries_per_cell_batch(j); v++)
          for(unsigned int i = 0; i < n_inner_face_batches_x; i++)
            (owning_class->*face_operation)(*this, dst, src, ID(i, j * v_len + v, i1++, ID::SpaceType::X));
    }
    for(unsigned int j = 0; j < n_inner_face_batches_v; j++)
    {
      ScopedTimerWrapper timer(timers, "face_loop_v");
      for(unsigned int v = 0; v < matrix_free_v.n_active_entries_per_face_batch(j); v++)
        for(unsigned int i = 0; i < n_cell_batches_x; i++)
          (owning_class->*face_operation)(*this, dst, src, ID(i, j * v_len + v, i1++, ID::SpaceType::V));
    }
      
    // ... and continue to loop over all boundary faces
    {
      ScopedTimerWrapper timer(timers, "boundary_loop_x");
      for(unsigned int j = 0; j < n_cell_batches_v; j++)
        for(unsigned int v = 0; v < matrix_free_v.n_active_entries_per_cell_batch(j); v++)
          for(unsigned int i = n_inner_face_batches_x; i < n_inner_or_boundary_face_batches_x; i++)
            (owning_class->*boundary_operation)(*this, dst, src, ID(i, j * v_len + v, i2++, ID::SpaceType::X));
    }
    {
      ScopedTimerWrapper timer(timers, "boundary_loop_v");
      for(unsigned int j = n_inner_face_batches_v; j < n_inner_or_boundary_face_batches_v; j++)
        for(unsigned int v = 0; v < matrix_free_v.n_active_entries_per_face_batch(j); v++)
          for(unsigned int i = 0; i < n_cell_batches_x; i++)
            (owning_class->*boundary_operation)(*this, dst, src, ID(i, j * v_len + v, i2++, ID::SpaceType::V));
    }
    // clang-format on

    if (dst_vector_face_access == DataAccessOnFaces::values)
      {
        ScopedTimerWrapper timer(timers, "compress");
        partitioner->compress_start(dst.begin(), dst.other_values());
        partitioner->compress_finish(dst.begin(), dst.other_values());
      }
    else
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  const MPI_Comm &
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::get_communicator()
    const
  {
    return comm;
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  dealii::types::boundary_id
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::get_boundary_id(
    const ID macro_face) const
  {
    unsigned int v_len = VectorizedArrayTypeV::size();

    if (macro_face.type == ID::SpaceType::X)
      return matrix_free_x.get_boundary_id(macro_face.x);
    else
      return matrix_free_v.get_boundary_id(macro_face.v / v_len);
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  bool
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::is_ecl_supported()
    const
  {
    return use_ecl;
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  bool
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
    are_ghost_faces_supported() const
  {
    return do_ghost_faces;
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  const dealii::MatrixFree<
    dim_x,
    Number,
    typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
      VectorizedArrayTypeX> &
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::get_matrix_free_x()
    const
  {
    return matrix_free_x;
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  const dealii::MatrixFree<
    dim_v,
    Number,
    typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
      VectorizedArrayTypeV> &
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::get_matrix_free_v()
    const
  {
    return matrix_free_v;
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  const internal::MatrixFreeFunctions::DoFInfo &
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::get_dof_info() const
  {
    return dof_info;
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  const internal::MatrixFreeFunctions::FaceInfo &
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::get_face_info() const
  {
    return face_info;
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  const internal::MatrixFreeFunctions::ShapeInfo<Number> &
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::get_shape_info() const
  {
    return shape_info;
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  const internal::MatrixFreeFunctions::VectorReaderWriter<Number> &
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::get_read_writer() const
  {
    return *reader_writer;
  }

} // namespace hyperdeal
