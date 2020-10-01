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

#include <hyper.deal/base/mpi_tags.h>
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
      // general (cached) info
      unsigned int max_batch_size;
      unsigned int n_cell_batches;

      // cell information
      std::vector<unsigned char> cells_fill; // for each macro cell
      std::vector<CellInfo>      cells;      // for each lane
      std::vector<unsigned int>  cells_lid;  //

      // face information in context of FCL
      std::vector<unsigned char> faces_fill;       // for each macro face
      std::vector<unsigned int>  interior_face_no; //
      std::vector<unsigned int>  exterior_face_no; //
      std::vector<unsigned int>  face_orientation; //
      std::vector<CellInfo>      cells_interior;   // for each lane
      std::vector<CellInfo>      cells_exterior;   //

      // face information in context of ECL
      std::vector<CellInfo>     cells_exterior_ecl;   // for each lane
      std::vector<unsigned int> exterior_face_no_ecl; //
      std::vector<unsigned int> face_orientation_ecl; //

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

    /**
     * Helper class to create a list of ghost faces.
     */
    struct GlobalCellInfoProcessor
    {
      /**
       * Constructor.
       */
      GlobalCellInfoProcessor(const GlobalCellInfo &info)
        : info(info)
      {}

      /**
       * Compute ghost faces.
       */
      std::vector<FaceInfo>
      get_ghost_faces(const int dim, const bool ecl = false) const
      {
        return get_ghost_faces(get_local_range(), dim, ecl);
      }

    private:
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
                             info.cells_exterior_ecl.size(),
                           dealii::ExcMessage("Out of range!"));

                    const unsigned int index =
                      i * info.max_batch_size * 2 * dim +
                      d * info.max_batch_size + v;

                    const auto cell_info   = info.cells_exterior_ecl[index];
                    const unsigned int gid = cell_info.gid;
                    if (gid == dealii::numbers::invalid_unsigned_int)
                      continue;
                    if (gid < i_min || i_max < gid)
                      ghosts_faces.emplace_back(
                        gid, cell_info.rank, info.exterior_face_no_ecl[index]);
                  }
          }
        else
          {
            for (unsigned int i = 0; i < info.interior_face_no.size(); i++)
              for (unsigned int v = 0; v < info.faces_fill[i]; v++)
                {
                  const auto cell_info =
                    info.cells_interior[i * info.max_batch_size + v];
                  const auto gid = cell_info.gid;

                  Assert(gid != dealii::numbers::invalid_dof_index,
                         dealii::StandardExceptions::ExcInternalError());

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
                  const auto gid = cell_info.gid;

                  Assert(gid != dealii::numbers::invalid_dof_index,
                         dealii::StandardExceptions::ExcInternalError());

                  if (gid < i_min || i_max < gid)
                    ghosts_faces.emplace_back(gid,
                                              cell_info.rank,
                                              info.exterior_face_no[i]);
                }
          }

        return ghosts_faces;
      }

      const GlobalCellInfo &info;
    };


    template <int dim, typename Number, typename VectorizedArrayType>
    void
    collect_global_cell_info(
      const dealii::MatrixFree<dim, Number, VectorizedArrayType> &data,
      GlobalCellInfo &                                            info)
    {
      // 1) create function to be able translate local cell ids to global ones
      // and get the rank of the owning process
      //
      // TODO: replace by global_active_cell_index once
      // https://github.com/dealii/dealii/pull/10490 is merged
      dealii::FE_DGQ<dim> fe(0);
      // ... distribute degrees of freedoms
      dealii::DoFHandler<dim> dof_handler(
        data.get_dof_handler().get_triangulation());
      dof_handler.distribute_dofs(fe);

      const auto cell_to_gid = [&](const auto &cell) {
        typename dealii::DoFHandler<dim>::level_cell_accessor dof_cell(
          &data.get_dof_handler().get_triangulation(),
          cell->level(),
          cell->index(),
          &dof_handler);

        std::vector<dealii::types::global_dof_index> indices(1);
        dof_cell.get_dof_indices(indices);
        return CellInfo(indices[0], dof_cell.subdomain_id());
      };

      // 2) allocate memory
      const unsigned int v_len = VectorizedArrayType::size();

      // ... general info
      info.max_batch_size = v_len;
      info.n_cell_batches = data.n_cell_batches();

      // ... cells
      info.cells.resize(v_len *
                        (data.n_cell_batches() + data.n_ghost_cell_batches()));
      info.cells_exterior_ecl.resize(dealii::GeometryInfo<dim>::faces_per_cell *
                                       v_len * data.n_cell_batches(),
                                     {dealii::numbers::invalid_unsigned_int,
                                      dealii::numbers::invalid_unsigned_int});
      info.cells_lid.resize(
        v_len * (data.n_cell_batches() + data.n_ghost_cell_batches()));
      info.cells_interior.resize(
        v_len * (data.n_inner_face_batches() + data.n_boundary_face_batches()));
      info.cells_exterior.resize(v_len * data.n_inner_face_batches());

      // ... fill
      info.cells_fill.resize(data.n_cell_batches() +
                             data.n_ghost_cell_batches());
      info.faces_fill.resize(data.n_inner_face_batches() +
                             data.n_boundary_face_batches());

      // ... face_no
      info.interior_face_no.resize(data.n_inner_face_batches() +
                                   data.n_boundary_face_batches());
      info.exterior_face_no.resize(data.n_inner_face_batches());
      info.exterior_face_no_ecl.resize(
        dealii::GeometryInfo<dim>::faces_per_cell * v_len *
        data.n_cell_batches());

      // ... face_orientation
      info.face_orientation.resize(data.n_inner_face_batches() +
                                   data.n_boundary_face_batches());
      info.face_orientation_ecl.resize(
        dealii::GeometryInfo<dim>::faces_per_cell * v_len *
        data.n_cell_batches());

      // 3) collect info by looping over local cells and ...
      for (unsigned int cell = 0;
           cell < data.n_cell_batches() + data.n_ghost_cell_batches();
           cell++)
        {
          info.cells_fill[cell] = data.n_components_filled(cell);

          // loop over all cells in macro cells and fill data structures
          unsigned int v = 0;
          for (; v < data.n_components_filled(cell); v++)
            {
              const auto c_it = data.get_cell_iterator(cell, v);

              // global id
              info.cells[cell * v_len + v] = cell_to_gid(c_it);

              // local id -> for data access
              //
              // warning: we assume that ghost cells have the same number
              // of unknowns as interior cells
              info.cells_lid[cell * v_len + v] =
                data.get_dof_info(/*TODO*/)
                  .dof_indices_contiguous[2][cell * v_len + v] /
                data.get_dofs_per_cell();

              // for interior cells ...
              if (cell < data.n_cell_batches())
                // ... loop over all its faces
                for (unsigned int face_no = 0;
                     face_no < dealii::GeometryInfo<dim>::faces_per_cell;
                     face_no++)
                  {
                    const unsigned int n_index =
                      cell * v_len * dealii::GeometryInfo<dim>::faces_per_cell +
                      v_len * face_no + v;

                    // on boundary faces: nothing to do so fill with invalid
                    // values
                    if (!c_it->has_periodic_neighbor(face_no) &&
                        c_it->at_boundary(face_no))
                      {
                        info.cells_exterior_ecl[n_index].gid  = -1;
                        info.cells_exterior_ecl[n_index].rank = -1;
                        info.exterior_face_no_ecl[n_index]    = -1;
                        info.face_orientation_ecl[n_index]    = -1;
                        continue;
                      }

                    // .. and collect the neighbors for ECL with the following
                    // information: 1) global id
                    info.cells_exterior_ecl[n_index] =
                      cell_to_gid(c_it->neighbor_or_periodic_neighbor(face_no));

                    // 2) face number and face orientation
                    //
                    // note: this is a modified copy and merged version of the
                    // methods dealii::FEFaceEvaluation::compute_face_no_data()
                    // and dealii::FEFaceEvaluation::compute_face_orientations()
                    // so that we don't have to use here FEEFaceEvaluation
                    {
                      const unsigned int cell_this =
                        cell * VectorizedArrayType::size() + v;

                      const unsigned int face_index =
                        data.get_cell_and_face_to_plain_faces()(cell,
                                                                face_no,
                                                                v);

                      Assert(face_index !=
                               dealii::numbers::invalid_unsigned_int,
                             dealii::StandardExceptions::ExcNotInitialized());

                      const auto &faces = data.get_face_info(
                        face_index / VectorizedArrayType::size());

                      const auto cell_m =
                        faces.cells_interior[face_index %
                                             VectorizedArrayType::size()];

                      const bool is_interior_face =
                        (cell_m != cell_this) ||
                        ((cell_m == cell_this) &&
                         (face_no !=
                          data
                            .get_face_info(face_index /
                                           VectorizedArrayType::size())
                            .interior_face_no));

                      info.exterior_face_no_ecl[n_index] =
                        is_interior_face ? faces.interior_face_no :
                                           faces.exterior_face_no;

                      if (dim == 3)
                        {
                          const bool fo_interior_face =
                            faces.face_orientation >= 8;

                          const unsigned int face_orientation =
                            faces.face_orientation % 8;

                          static const std::array<unsigned int, 8> table{
                            {0, 1, 2, 3, 6, 5, 4, 7}};

                          info.face_orientation_ecl[n_index] =
                            (is_interior_face != fo_interior_face) ?
                              table[face_orientation] :
                              face_orientation;
                        }
                      else
                        info.face_orientation_ecl[n_index] = -1;
                    }
                  }
            }
        }

      // ... interior faces (filled lanes, face_no_m/_p, gid_m/_p)
      for (unsigned int face = 0; face < data.n_inner_face_batches(); ++face)
        {
          // number of filled lanes
          info.faces_fill[face] = data.n_active_entries_per_face_batch(face);

          // face number
          info.interior_face_no[face] =
            data.get_face_info(face).interior_face_no;
          info.exterior_face_no[face] =
            data.get_face_info(face).exterior_face_no;

          // face orientation
          info.face_orientation[face] =
            data.get_face_info(face).face_orientation;

          // process cells in batch
          for (unsigned int v = 0;
               v < data.n_active_entries_per_face_batch(face);
               ++v)
            {
              const auto cell_m = data.get_face_info(face).cells_interior[v];
              info.cells_interior[face * v_len + v] = cell_to_gid(
                data.get_cell_iterator(cell_m / v_len, cell_m % v_len));

              const auto cell_p = data.get_face_info(face).cells_exterior[v];
              info.cells_exterior[face * v_len + v] = cell_to_gid(
                data.get_cell_iterator(cell_p / v_len, cell_p % v_len));
            }
        }

      // ... boundary faces (filled lanes, face_no_m, gid_m)
      for (unsigned int face = data.n_inner_face_batches();
           face < data.n_inner_face_batches() + data.n_boundary_face_batches();
           ++face)
        {
          info.faces_fill[face] = data.n_active_entries_per_face_batch(face);

          info.interior_face_no[face] =
            data.get_face_info(face).interior_face_no;

          for (unsigned int v = 0;
               v < data.n_active_entries_per_face_batch(face);
               ++v)
            {
              const auto cell_m = data.get_face_info(face).cells_interior[v];
              info.cells_interior[face * v_len + v] = cell_to_gid(
                data.get_cell_iterator(cell_m / v_len, cell_m % v_len));
            }
        }
    }


    /**
     * A class to translate the tensor product of global cell ids to a
     * single global id. Processes are enumerated lexicographically and cells
     * are enumerated lexicographically and continuously within a process.
     */
    class GlobaleCellIDTranslator
    {
    public:
      /**
       * Constructor.
       */
      GlobaleCellIDTranslator(const GlobalCellInfo &info_x,
                              const GlobalCellInfo &info_v,
                              const MPI_Comm        comm_x,
                              const MPI_Comm        comm_v)
      {
        // determine the first cell of each rank in x-space
        {
          // allocate memory
          n1.resize(dealii::Utilities::MPI::n_mpi_processes(comm_x) + 1);

          // number of locally owned cells
          const unsigned int n_local = info_x.compute_n_cells();

          // gather the number of cells of all processes
          MPI_Allgather(
            &n_local, 1, MPI_UNSIGNED, n1.data() + 1, 1, MPI_UNSIGNED, comm_x);

          // perform prefix sum
          for (unsigned int i = 0; i < n1.size() - 1; i++)
            n1[i + 1] += n1[i];
        }

        // the same for v-space
        {
          n2.resize(dealii::Utilities::MPI::n_mpi_processes(comm_v) + 1);
          unsigned int local = info_v.compute_n_cells();
          MPI_Allgather(
            &local, 1, MPI_UNSIGNED, n2.data() + 1, 1, MPI_UNSIGNED, comm_v);
          for (unsigned int i = 0; i < n2.size() - 1; i++)
            n2[i + 1] += n2[i];
        }
      }

      /**
       * Translate the tensor product of global cell IDs to a single global ID.
       */
      CellInfo
      translate(const CellInfo &id1, const CellInfo &id2) const
      {
        // extract needed information
        const unsigned int gid1  = id1.gid;
        const unsigned int rank1 = id1.rank;
        const unsigned int gid2  = id2.gid;
        const unsigned int rank2 = id2.rank;

        Assert(rank1 != static_cast<unsigned int>(-1),
               dealii::StandardExceptions::ExcNotImplemented());
        Assert(rank2 != static_cast<unsigned int>(-1),
               dealii::StandardExceptions::ExcNotImplemented());

        AssertIndexRange(rank1 + 1, n1.size());
        AssertIndexRange(rank2 + 1, n2.size());

        // 1) determine local IDs
        const unsigned int lid1 = gid1 - n1[rank1];
        const unsigned int lid2 = gid2 - n2[rank2];

        // 2) determine local ID by taking taking tensor product
        const unsigned int lid = lid1 + lid2 * (n1[rank1 + 1] - n1[rank1]);

        // 3) add offset to local ID to get global ID
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

      // helper function to create the global cell id of a tensor-product cell
      GlobaleCellIDTranslator translator(info_x, info_v, comm_x, comm_v);

      // helper function to create the tensor product of cells
      const auto process = [&](const unsigned int i_x,
                               const unsigned int i_v,
                               const unsigned int v_v) {
        unsigned int v_x = 0;
        for (; v_x < info_x.cells_fill[i_x]; v_x++)
          {
            const auto cell_x = info_x.cells[i_x * info_x.max_batch_size + v_x];
            const auto cell_y = info_v.cells[i_v * info_v.max_batch_size + v_v];
            info.cells.emplace_back(translator.translate(cell_x, cell_y));
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
                      const unsigned int index =
                        i_x * info_x.max_batch_size * 2 * dim_x +
                        d * info_x.max_batch_size + v_x;
                      const auto cell_x = info_x.cells_exterior_ecl[index];
                      const auto cell_y =
                        info_v.cells[i_v * info_v.max_batch_size + v_v];

                      // on boundary faces: nothing to do so fill with invalid
                      // values
                      if (cell_x.gid == static_cast<decltype(cell_x.gid)>(-1) ||
                          cell_y.gid == static_cast<decltype(cell_y.gid)>(-1))
                        {
                          info.cells_exterior_ecl.emplace_back(-1, -1);
                          info.exterior_face_no_ecl.emplace_back(-1);
                          info.face_orientation_ecl.emplace_back(-1);
                          continue;
                        }

                      info.cells_exterior_ecl.emplace_back(
                        translator.translate(cell_x, cell_y));
                      info.exterior_face_no_ecl.emplace_back(
                        info_x.exterior_face_no_ecl[index]);
                      info.face_orientation_ecl.emplace_back(
                        info_x.face_orientation_ecl[index]); // TODO?
                    }
                  for (; v_x < info_x.max_batch_size; v_x++)
                    {
                      info.cells_exterior_ecl.emplace_back(-1, -1);
                      info.exterior_face_no_ecl.emplace_back(-1);
                      info.face_orientation_ecl.emplace_back(-1);
                    }
                }

              for (int d = 0; d < 2 * dim_v; d++)
                {
                  unsigned int v_x = 0;
                  for (; v_x < info_x.cells_fill[i_x]; v_x++)
                    {
                      const unsigned index =
                        i_v * info_v.max_batch_size * 2 * dim_v +
                        d * info_v.max_batch_size + v_v;
                      const auto cell_x =
                        info_x.cells[i_x * info_x.max_batch_size + v_x];
                      const auto cell_y = info_v.cells_exterior_ecl[index];

                      // on boundary faces: nothing to do so fill with invalid
                      // values
                      if (cell_x.gid == static_cast<decltype(cell_x.gid)>(-1) ||
                          cell_y.gid == static_cast<decltype(cell_y.gid)>(-1))
                        {
                          info.cells_exterior_ecl.emplace_back(-1, -1);
                          info.exterior_face_no_ecl.emplace_back(-1);
                          info.face_orientation_ecl.emplace_back(-1);
                          continue;
                        }

                      info.cells_exterior_ecl.emplace_back(
                        translator.translate(cell_x, cell_y));
                      info.exterior_face_no_ecl.emplace_back(
                        info_v.exterior_face_no_ecl[index] + 2 * dim_x);
                      info.face_orientation_ecl.emplace_back(
                        info_v.face_orientation_ecl[index]); // TODO?
                    }
                  for (; v_x < info_x.max_batch_size; v_x++)
                    {
                      info.cells_exterior_ecl.emplace_back(-1, -1);
                      info.exterior_face_no_ecl.emplace_back(-1);
                      info.face_orientation_ecl.emplace_back(-1);
                    }
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
        const unsigned int n_face_batches_x = info_x.exterior_face_no.size();
        const unsigned int n_face_batches_v = info_v.exterior_face_no.size();

        const unsigned int n_face_batches_x_all =
          info_x.interior_face_no.size();
        const unsigned int n_face_batches_v_all =
          info_v.interior_face_no.size();
        // interior faces (face x cell):
        for (unsigned int i_v = 0; i_v < n_cell_batches_v; i_v++)
          for (unsigned int v_v = 0; v_v < info_v.cells_fill[i_v]; v_v++)
            for (unsigned int i_x = 0; i_x < n_face_batches_x; i_x++)
              {
                unsigned int v_x = 0;
                for (; v_x < info_x.faces_fill[i_x]; v_x++)
                  {
                    const auto cell_y =
                      info_v.cells[i_v * info_v.max_batch_size + v_v];
                    info.cells_interior.emplace_back(translator.translate(
                      info_x.cells_interior[i_x * info_x.max_batch_size + v_x],
                      cell_y));
                    info.cells_exterior.emplace_back(translator.translate(
                      info_x.cells_exterior[i_x * info_x.max_batch_size + v_x],
                      cell_y));
                  }
                for (; v_x < info_x.max_batch_size; v_x++)
                  {
                    info.cells_interior.emplace_back(-1, -1);
                    info.cells_exterior.emplace_back(-1, -1);
                  }

                info.faces_fill.push_back(info_x.faces_fill[i_x]);

                info.interior_face_no.push_back(info_x.interior_face_no[i_x]);
                info.exterior_face_no.push_back(info_x.exterior_face_no[i_x]);
                info.face_orientation.push_back(info_x.face_orientation[i_x]);
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
                    info.cells_interior.emplace_back(translator.translate(
                      cell_x,
                      info_v
                        .cells_interior[i_v * info_v.max_batch_size + v_v]));
                    info.cells_exterior.emplace_back(translator.translate(
                      cell_x,
                      info_v
                        .cells_exterior[i_v * info_v.max_batch_size + v_v]));
                  }
                for (; v_x < info_x.max_batch_size; v_x++)
                  {
                    info.cells_interior.emplace_back(-1, -1);
                    info.cells_exterior.emplace_back(-1, -1);
                  }

                info.faces_fill.push_back(info_x.cells_fill[i_x]);

                info.interior_face_no.push_back(info_v.interior_face_no[i_v] +
                                                2 * dim_x);
                info.exterior_face_no.push_back(info_v.exterior_face_no[i_v] +
                                                2 * dim_x);
                info.face_orientation.push_back(info_v.face_orientation[i_v]);
              }

        for (unsigned int i_v = 0; i_v < n_cell_batches_v; i_v++)
          for (unsigned int v_v = 0; v_v < info_v.cells_fill[i_v]; v_v++)
            for (unsigned int i_x = n_face_batches_x;
                 i_x < n_face_batches_x_all;
                 i_x++)
              {
                unsigned int v_x = 0;
                for (; v_x < info_x.faces_fill[i_x]; v_x++)
                  {
                    const auto cell_x =
                      info_x.cells_interior[i_x * info_x.max_batch_size + v_x];
                    const auto cell_y =
                      info_v.cells[i_v * info_v.max_batch_size + v_v];
                    info.cells_interior.emplace_back(
                      translator.translate(cell_x, cell_y));
                  }
                for (; v_x < info_x.max_batch_size; v_x++)
                  info.cells_interior.emplace_back(-1, -1);

                info.faces_fill.push_back(info_x.faces_fill[i_x]);

                info.interior_face_no.push_back(info_x.interior_face_no[i_x]);
                info.face_orientation.push_back(info_x.face_orientation[i_x]);
              }

        // interior faces (cell x face):
        for (unsigned int i_v = n_face_batches_v; i_v < n_face_batches_v_all;
             i_v++)
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
                    info.cells_interior.emplace_back(
                      translator.translate(cell_x, cell_y));
                  }
                for (; v_x < info_x.max_batch_size; v_x++)
                  info.cells_interior.emplace_back(-1, -1);

                info.faces_fill.push_back(info_x.cells_fill[i_x]);

                info.interior_face_no.push_back(info_v.interior_face_no[i_v] +
                                                2 * dim_x);
                info.face_orientation.push_back(info_v.face_orientation[i_v]);
              }
      }
    }
  } // namespace internal


  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::MatrixFree(
    const MPI_Comm comm,
    const MPI_Comm comm_sm,
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

    AssertDimension(matrix_free_x.get_shape_info().n_components, 1);
    AssertDimension(matrix_free_v.get_shape_info().n_components, 1);
    AssertDimension(matrix_free_x.get_shape_info().data.front().fe_degree,
                    matrix_free_v.get_shape_info().data.front().fe_degree);

    const int dim = dim_x + dim_v;

    // set up shape_info
    this->shape_info.template reinit<dim_x, dim_v>(
      matrix_free_x.get_shape_info().data.front().fe_degree);

    // collect (global) information of each macro cell in phase space
    const auto info = [&]() {
      internal::GlobalCellInfo info_x, info_v, info;

      // collect (global) information of each macro cell in x-space
      internal::collect_global_cell_info(matrix_free_x, info_x);

      // collect (global) information of each macro cell in v-space
      internal::collect_global_cell_info(matrix_free_v, info_v);

      // create tensor product
      internal::combine_global_cell_info(
        matrix_free_x.get_task_info().communicator,
        matrix_free_v.get_task_info().communicator,
        info_x,
        info_v,
        info,
        dim_x,
        dim_v);

      return info;
    }();

    // set up partitioner
    this->partitioner = [&] {
      AssertThrow(do_ghost_faces,
                  dealii::StandardExceptions::ExcNotImplemented());

      auto partitioner =
        new internal::MatrixFreeFunctions::Partitioner(shape_info);

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
        // a) get ghost faces
        auto ghost_faces =
          internal::GlobalCellInfoProcessor(info).get_ghost_faces(
            dim, this->use_ecl);

        // b) sort them
        std::sort(ghost_faces.begin(),
                  ghost_faces.end(),
                  [](const auto &a, const auto &b) {
                    // according to global id of corresponding cell ...
                    if (a.gid < b.gid)
                      return true;

                    // ... and face number
                    if (a.gid == b.gid && a.no < b.no)
                      return true;

                    return false;
                  });

        // c) group them
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

      // actually setup partitioner
      partitioner->reinit(local_list, ghost_list, comm, comm_sm, do_buffering);

      return std::shared_ptr<dealii::LinearAlgebra::SharedMPI::PartitionerBase>(
        partitioner);
    }();


    // set up rest of dof_info and face_info (1)
    {
      auto &n_vectorization_lanes_filled =
        dof_info.n_vectorization_lanes_filled;
      auto &dof_indices_contiguous = dof_info.dof_indices_contiguous;
      auto &no_faces               = face_info.no_faces;
      auto &face_orientations      = face_info.face_orientations;

      // 1) collect gids (dof_indices) according to vectorization
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
                  .cells_exterior_ecl
                    [i * info.max_batch_size *
                       dealii::GeometryInfo<dim>::faces_per_cell +
                     d * info.max_batch_size + v]
                  .gid);
      }

      // 2) collect filled lanes
      {
        for (unsigned int i = 0; i < info.interior_face_no.size(); i++)
          n_vectorization_lanes_filled[0].push_back(info.faces_fill[i]);

        for (unsigned int i = 0; i < info.exterior_face_no.size(); i++)
          n_vectorization_lanes_filled[1].push_back(info.faces_fill[i]);

        for (unsigned int i = 0; i < info.n_cell_batches; i++)
          n_vectorization_lanes_filled[2].push_back(info.cells_fill[i]);

        if (this->use_ecl) // filled only so that the code is more generic
                           // cleared later on!
          for (unsigned int i = 0; i < info.n_cell_batches; i++)
            for (unsigned int d = 0;
                 d < dealii::GeometryInfo<dim>::faces_per_cell;
                 d++)
              n_vectorization_lanes_filled[3].push_back(info.cells_fill[i]);
      }

      // 3) collect face numbers
      {
        no_faces[0] = info.interior_face_no;
        no_faces[1] = info.exterior_face_no;
        no_faces[3] = info.exterior_face_no_ecl;
      }

      // 3) collect face orientations
      {
        face_orientations[0] = info.face_orientation;
        face_orientations[3] = info.face_orientation_ecl;
      }
    }


    // set up rest of dof_info and face_info (2)
    {
      // given information
      const auto &vectorization_length = dof_info.n_vectorization_lanes_filled;
      const auto &dof_indices_contiguous = dof_info.dof_indices_contiguous;
      const auto &no_faces               = face_info.no_faces;
      const auto &face_orientations      = face_info.face_orientations;

      const auto &maps =
        dynamic_cast<const internal::MatrixFreeFunctions::Partitioner *>(
          partitioner.get())
          ->get_maps();
      const auto &maps_ghost =
        dynamic_cast<const internal::MatrixFreeFunctions::Partitioner *>(
          partitioner.get())
          ->get_maps_ghost();

      static const int v_len = VectorizedArrayType::size();

      // to be computed
      auto &cell_ptrs = dof_info.dof_indices_contiguous_ptr;
      auto &face_type = face_info.face_type;
      auto &face_all  = face_info.face_all;

      // allocate memory
      for (unsigned int i = 0; i < 4; i++)
        cell_ptrs[i].resize(dof_indices_contiguous[i].size());

      for (unsigned int i = 0; i < 4; i++)
        face_type[i].resize(i == 2 ? 0 : dof_indices_contiguous[i].size());

      for (unsigned int i = 0; i < 4; i++)
        face_all[i].resize(i == 2 ? 0 : vectorization_length[i].size());

      // process faces: cell_ptrs and face_type
      for (unsigned int i = 0; i < 4; i++)
        {
          if (i == 2)
            continue; // nothing to do for cells - it is done later

          for (unsigned int j = 0; j < vectorization_length[i].size(); j++)
            for (unsigned int v = 0; v < vectorization_length[i][j]; v++)
              {
                const unsigned int l = j * v_len + v;
                Assert(l < dof_indices_contiguous[i].size(),
                       dealii::StandardExceptions::ExcMessage(
                         "Size of gid does not match."));
                const unsigned int gid_this = dof_indices_contiguous[i][l];

                // for boundary faces nothing has to be done
                if (gid_this == dealii::numbers::invalid_unsigned_int)
                  continue;

                const auto ptr1 = maps_ghost.find(
                  {gid_this,
                   do_ghost_faces ? no_faces[i][i == 3 ? l : j] :
                                    dealii::numbers::invalid_unsigned_int});

                if (ptr1 != maps_ghost.end())
                  {
                    cell_ptrs[i][l] = {ptr1->second.first, ptr1->second.second};
                    face_type[i][l] = true; // ghost face
                    continue;
                  }

                const auto ptr2 = maps.find(gid_this);

                if (ptr2 != maps.end())
                  {
                    cell_ptrs[i][l] = {ptr2->second.first, ptr2->second.second};
                    face_type[i][l] = false; // cell is part of sm
                    continue;
                  }

                AssertThrow(false,
                            dealii::StandardExceptions::ExcMessage(
                              "Cell not found!"));
              }
        }

      // process faces: face_all
      for (unsigned int i = 0; i < 4; i++)
        {
          if (i == 2)
            continue; // nothing to do for cells - it is done later

          for (unsigned int j = 0; j < vectorization_length[i].size(); j++)
            {
              bool temp = true;
              for (unsigned int v = 0; v < vectorization_length[i][j]; v++)
                temp &=
                  (face_type[i][j * v_len] == face_type[i][j * v_len + v]) &&
                  (i == 3 ?
                     ((no_faces[i][j * v_len] == no_faces[i][j * v_len + v]) &&
                      (face_orientations[i][j * v_len] ==
                       face_orientations[i][j * v_len + v])) :
                     true);
              face_all[i][j] = temp;
            }
        }

      // process cells
      for (unsigned int j = 0; j < vectorization_length[2].size(); j++)
        for (unsigned int v = 0; v < vectorization_length[2][j]; v++)
          {
            const unsigned int l        = j * v_len + v;
            const unsigned int gid_this = dof_indices_contiguous[2][l];
            const auto         ptr      = maps.find(gid_this);

            cell_ptrs[2][l] = {ptr->second.first, ptr->second.second};
          }

      // clear vector since it is not needed anymore, since the values
      // are the same as for cells
      dof_info.n_vectorization_lanes_filled[3].clear();
    }

    // partitions for ECL
    if (this->use_ecl)
      {
        for (auto &partition : partitions)
          partition.clear();

        const unsigned int v_len = VectorizedArrayTypeV::size();
        const auto my_rank = dealii::Utilities::MPI::this_mpi_process(comm);
        const auto sm_ranks_vec = mpi::procs_of_sm(comm, comm_sm);
        const std::set<unsigned int> sm_ranks(sm_ranks_vec.begin(),
                                              sm_ranks_vec.end());

        const unsigned int n_cell_batches_x = matrix_free_x.n_cell_batches();
        const unsigned int n_cell_batches_v = matrix_free_v.n_cell_batches();

        // 0: no overlapping
        const unsigned int overlapping_level =
          additional_data.overlapping_level;

        for (unsigned int j = 0, i0 = 0; j < n_cell_batches_v; j++)
          for (unsigned int v = 0;
               v < matrix_free_v.n_active_entries_per_cell_batch(j);
               v++)
            for (unsigned int i = 0; i < n_cell_batches_x; i++, i0++)
              {
                unsigned int flag = 0;
                for (unsigned int d = 0;
                     d < dealii::GeometryInfo<dim>::faces_per_cell;
                     d++)
                  for (unsigned int v = 0;
                       v < dof_info.n_vectorization_lanes_filled[2][i0];
                       v++)
                    {
                      const auto gid =
                        info
                          .cells_exterior_ecl
                            [i0 * info.max_batch_size *
                               dealii::GeometryInfo<dim>::faces_per_cell +
                             d * info.max_batch_size + v]
                          .rank;

                      if (overlapping_level >= 1 && gid == my_rank)
                        flag = std::max(flag, 0u);
                      else if (overlapping_level == 2 &&
                               sm_ranks.find(gid) != sm_ranks.end())
                        flag = std::max(flag, 1u);
                      else
                        flag = std::max(flag, 2u);
                    }

                partitions[flag].emplace_back(i, j * v_len + v, i0);
              }

        if (dealii::Utilities::MPI::this_mpi_process(comm) == 0)
          std::cout << partitions[0].size() << " " << partitions[1].size()
                    << " " << partitions[2].size() << " " << std::endl;
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

    AssertThrow((partitioner != nullptr),
                dealii::ExcMessage("Partitioner has not been initialized!"));

    // setup vector
    vec.reinit(std::make_shared<dealii::Utilities::MPI::Partitioner>(),
               partitioner,
               do_ghosts);

    // zero out values
    if (zero_out_values)
      vec = 0.0;

    // perform test ghost value update (working for ECL/FCL)
    if (zero_out_values && do_ghosts)
      {
        vec.zero_out_ghosts();

        dealii::AlignedVector<Number> buffer;
        std::vector<MPI_Request>      requests;

        this->partitioner->export_to_ghosted_array_start(
          0, vec.begin(), vec.other_values(), buffer, requests);

        this->partitioner->export_to_ghosted_array_finish(vec.begin(),
                                                          vec.other_values(),
                                                          requests);

        vec.zero_out_ghosts();
      }

    // perform test compression (working for FCL)
    if (zero_out_values && do_ghosts && !use_ecl)
      {
        vec.compress(dealii::VectorOperation::values::add);

        dealii::AlignedVector<Number> buffer;
        std::vector<MPI_Request>      requests;

        this->partitioner->import_from_ghosted_array_start(
          dealii::VectorOperation::values::add,
          0,
          vec.begin(),
          vec.other_values(),
          buffer,
          requests);

        this->partitioner->import_from_ghosted_array_finish(
          dealii::VectorOperation::values::add,
          vec.begin(),
          vec.other_values(),
          buffer,
          requests);
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
    this->template loop_cell_centric<OutVector, InVector>(
      [&cell_operation, &owning_class](const MatrixFree &mf,
                                       OutVector &       dst,
                                       const InVector &  src,
                                       const ID          id) {
        (owning_class->*cell_operation)(mf, dst, src, id);
      },
      dst,
      src,
      src_vector_face_access,
      timers);
  }

  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  template <typename OutVector, typename InVector>
  void
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::loop_cell_centric(
    const std::function<
      void(const MatrixFree &, OutVector &, const InVector &, const ID)>
      &                     cell_operation,
    OutVector &             dst,
    const InVector &        src,
    const DataAccessOnFaces src_vector_face_access,
    Timers *                timers) const
  {
    AssertThrow(src_vector_face_access == DataAccessOnFaces::values ||
                  src_vector_face_access == DataAccessOnFaces::none,
                dealii::StandardExceptions::ExcNotImplemented());

    const auto part =
      dynamic_cast<const internal::MatrixFreeFunctions::Partitioner *>(
        partitioner.get());

    {
      ScopedTimerWrapper timer(timers, "loop");

      InVector &                    src_ = const_cast<InVector &>(src);
      dealii::AlignedVector<Number> buffer;
      std::vector<MPI_Request>      requests;

      // loop over all partitions
      for (unsigned int i = 0; i < this->partitions.size(); ++i)
        {
          // perform pre-processing step for partition
          if (src_vector_face_access == DataAccessOnFaces::values)
            {
              if (i == 0)
                {
                  if (timers != nullptr)
                    timers->operator[]("update_ghost_values_0").start();

                  // perform src.update_ghost_values_start()
                  part->export_to_ghosted_array_start(
                    0, src_.begin(), src_.other_values(), buffer, requests);

                  // zero out ghost of destination vector
                  if (dst.has_ghost_elements())
                    dst.zero_out_ghosts();
                }
              else if (i == 1)
                {
                  // ... src.update_ghost_values_finish() for shared-memory
                  // domain:
                  part->export_to_ghosted_array_finish_0(src_.begin(),
                                                         src_.other_values(),
                                                         requests);

                  if (timers != nullptr)
                    {
                      timers->operator[]("update_ghost_values_0").stop();
                      timers->operator[]("update_ghost_values_1").start();
                    }
                }
              else if (i == 2)
                {
                  // ... src.update_ghost_values_finish() for remote domain:
                  part->export_to_ghosted_array_finish_1(src_.begin(),
                                                         src_.other_values(),
                                                         requests);

                  if (timers != nullptr)
                    {
                      timers->operator[]("update_ghost_values_1").stop();
                      timers->operator[]("update_ghost_values_2").start();
                    }
                }
              else
                {
                  AssertThrow(false,
                              dealii::StandardExceptions::ExcNotImplemented());
                }
            }

          // loop over all cells in partition
          for (const auto &id : this->partitions[i])
            cell_operation(*this, dst, src, id);
        }
    }

    if (timers != nullptr)
      timers->operator[]("update_ghost_values_2").stop();

    if (do_buffering == false &&
        src_vector_face_access == DataAccessOnFaces::values)
      {
        ScopedTimerWrapper timer(timers, "barrier");
        part->sync();
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
    this->template loop<OutVector, InVector>(
      [&cell_operation, &owning_class](const MatrixFree &mf,
                                       OutVector &       dst,
                                       const InVector &  src,
                                       const ID          id) {
        (owning_class->*cell_operation)(mf, dst, src, id);
      },
      [&face_operation, &owning_class](const MatrixFree &mf,
                                       OutVector &       dst,
                                       const InVector &  src,
                                       const ID          id) {
        (owning_class->*face_operation)(mf, dst, src, id);
      },
      [&boundary_operation, &owning_class](const MatrixFree &mf,
                                           OutVector &       dst,
                                           const InVector &  src,
                                           const ID          id) {
        (owning_class->*boundary_operation)(mf, dst, src, id);
      },
      dst,
      src,
      dst_vector_face_access,
      src_vector_face_access,
      timers);
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  template <typename OutVector, typename InVector>
  void
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::loop(
    const std::function<
      void(const MatrixFree &, OutVector &, const InVector &, const ID)>
      &cell_operation,
    const std::function<
      void(const MatrixFree &, OutVector &, const InVector &, const ID)>
      &face_operation,
    const std::function<
      void(const MatrixFree &, OutVector &, const InVector &, const ID)>
      &                     boundary_operation,
    OutVector &             dst,
    const InVector &        src,
    const DataAccessOnFaces dst_vector_face_access,
    const DataAccessOnFaces src_vector_face_access,
    Timers *                timers) const
  {
    if (src_vector_face_access == DataAccessOnFaces::values)
      {
        ScopedTimerWrapper timer(timers, "update_ghost_values");

        src.update_ghost_values(); // TODO: use partitioner directly
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

    // clang-format off
  
    // loop over all cells
    {
      ScopedTimerWrapper timer(timers, "cell_loop");
      for(unsigned int j = 0; j < n_cell_batches_v; j++)
        for(unsigned int v = 0; v < matrix_free_v.n_active_entries_per_cell_batch(j); v++)
          for(unsigned int i = 0; i < n_cell_batches_x; i++)
            cell_operation(*this, dst, src, ID(i, j * v_len + v, i0++));
    }
  
    // loop over all inner faces ...
    {
      ScopedTimerWrapper timer(timers, "face_loop_x");
      for(unsigned int j = 0; j < n_cell_batches_v; j++)
        for(unsigned int v = 0; v < matrix_free_v.n_active_entries_per_cell_batch(j); v++)
          for(unsigned int i = 0; i < n_inner_face_batches_x; i++)
            face_operation(*this, dst, src, ID(i, j * v_len + v, i1++, ID::SpaceType::X));
    }
    for(unsigned int j = 0; j < n_inner_face_batches_v; j++)
    {
      ScopedTimerWrapper timer(timers, "face_loop_v");
      for(unsigned int v = 0; v < matrix_free_v.n_active_entries_per_face_batch(j); v++)
        for(unsigned int i = 0; i < n_cell_batches_x; i++)
          face_operation(*this, dst, src, ID(i, j * v_len + v, i1++, ID::SpaceType::V));
    }
      
    // ... and continue to loop over all boundary faces
    {
      ScopedTimerWrapper timer(timers, "boundary_loop_x");
      for(unsigned int j = 0; j < n_cell_batches_v; j++)
        for(unsigned int v = 0; v < matrix_free_v.n_active_entries_per_cell_batch(j); v++)
          for(unsigned int i = n_inner_face_batches_x; i < n_inner_or_boundary_face_batches_x; i++)
            boundary_operation(*this, dst, src, ID(i, j * v_len + v, i1++, ID::SpaceType::X));
    }
    {
      ScopedTimerWrapper timer(timers, "boundary_loop_v");
      for(unsigned int j = n_inner_face_batches_v; j < n_inner_or_boundary_face_batches_v; j++)
        for(unsigned int v = 0; v < matrix_free_v.n_active_entries_per_face_batch(j); v++)
          for(unsigned int i = 0; i < n_cell_batches_x; i++)
            boundary_operation(*this, dst, src, ID(i, j * v_len + v, i1++, ID::SpaceType::V));
    }
    // clang-format on

    if (dst_vector_face_access == DataAccessOnFaces::values)
      {
        ScopedTimerWrapper timer(timers, "compress");

        dst.compress(
          dealii::VectorOperation::add); // TODO: use partitioner directly
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
    if (macro_face.type == TensorID::SpaceType::X)
      return matrix_free_x.get_boundary_id(macro_face.x);
    else if (macro_face.type == TensorID::SpaceType::V)
      return matrix_free_v.get_boundary_id(macro_face.v /
                                           VectorizedArrayTypeV::size());

    Assert(false, dealii::StandardExceptions::ExcInternalError());

    return -1;
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  dealii::types::boundary_id
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
    get_faces_by_cells_boundary_id(const TensorID &   macro_cell,
                                   const unsigned int face_number) const
  {
    if (face_number < 2 * dim_x)
      {
        const auto bids =
          matrix_free_x.get_faces_by_cells_boundary_id(macro_cell.x,
                                                       face_number);

#ifdef DEBUG
        for (unsigned int v = 0;
             v < matrix_free_x.n_active_entries_per_cell_batch(macro_cell.x);
             ++v)
          AssertDimension(bids[0], bids[v]);
#endif

        return bids[0];
      }
    else if (face_number < 2 * dim_x + 2 * dim_v)
      {
        const auto bids =
          matrix_free_v.get_faces_by_cells_boundary_id(macro_cell.v,
                                                       face_number - dim_x * 2);

#ifdef DEBUG
        for (unsigned int v = 0;
             v < matrix_free_v.n_active_entries_per_cell_batch(macro_cell.v);
             ++v)
          AssertDimension(bids[0], bids[v]);
#endif

        return bids[0];
      }

    Assert(false, dealii::StandardExceptions::ExcInternalError());

    return -1;
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
  MemoryConsumption
  MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::memory_consumption()
    const
  {
    MemoryConsumption mem("matrix_free");
    mem.insert("partitioner", partitioner->memory_consumption());
    mem.insert("dof_info", dof_info.memory_consumption());
    mem.insert("face_info", face_info.memory_consumption());
    mem.insert("shape_info", shape_info.memory_consumption());

    return mem;
  }



  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  const std::shared_ptr<const dealii::LinearAlgebra::SharedMPI::PartitionerBase>
    &
    MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
      get_vector_partitioner() const
  {
    return partitioner;
  }

} // namespace hyperdeal
