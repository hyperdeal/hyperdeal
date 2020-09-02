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

#ifndef DEALII_LINEARALGEBRA_SHAREDMPI_PARTITIONER
#define DEALII_LINEARALGEBRA_SHAREDMPI_PARTITIONER

#include <hyper.deal/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/mpi_compute_index_owner_internal.h>
#include <deal.II/base/mpi_consensus_algorithms.h>
#include <deal.II/base/timer.h>

#ifdef DEAL_II_WITH_64BIT_INDICES
#  include <deal.II/base/mpi_consensus_algorithms.templates.h>
#endif

#include <deal.II/lac/la_sm_partitioner.h>

#include <hyper.deal/base/mpi.h>
#include <hyper.deal/base/mpi_tags.h>
#include <hyper.deal/matrix_free/shape_info.h>

#include <map>
#include <vector>


namespace hyperdeal
{
  namespace internal
  {
    namespace MatrixFreeFunctions
    {
      /**
       * Partitioner for discontinuous Galerkin discretizations, exploiting
       * shared memory.
       */
      class Partitioner
        : public dealii::LinearAlgebra::SharedMPI::PartitionerBase
      {
        const dealii::types::global_dof_index         dofs_per_cell;
        const dealii::types::global_dof_index         dofs_per_face;
        const std::vector<std::vector<unsigned int>> &face_to_cell_index_nodal;

      public:
        using RankType     = unsigned int;
        using LocalDoFType = unsigned int;
        using CellIdType   = dealii::types::global_dof_index;
        using FaceIdType   = std::pair<CellIdType, unsigned int>;

        /**
         * Constructor.
         */
        template <typename Number>
        Partitioner(const ShapeInfo<Number> &shape_info);

        /**
         * Constructor.
         *
         * @note Not implemented. Use the other reinit function.
         */
        void
        reinit(const dealii::IndexSet &is_locally_owned,
               const dealii::IndexSet &is_locally_ghost,
               const MPI_Comm &        communicator) override;

        /**
         * Initialize partitioner with a list of locally owned cells and
         * a list of ghost faces (cell and face no).
         */
        void
        reinit(const std::vector<dealii::types::global_dof_index> local_cells,
               const std::vector<std::pair<dealii::types::global_dof_index,
                                           std::vector<unsigned int>>>
                              local_ghost_faces,
               const MPI_Comm comm,
               const MPI_Comm comm_sm,
               const bool     do_buffering);

        /**
         * Start to export to ghost array.
         */
        void
        export_to_ghosted_array_start(
          const unsigned int             communication_channel,
          double *const                  data_this,
          const std::vector<double *> &  data_others,
          dealii::AlignedVector<double> &buffer,
          std::vector<MPI_Request> &     requests) const override;

        /**
         * Finish to export to ghost array.
         */
        void
        export_to_ghosted_array_finish(
          double *const                data_this,
          const std::vector<double *> &data_others,
          std::vector<MPI_Request> &   requests) const override;

        /**
         * Start to import from ghost array.
         */
        void
        import_from_ghosted_array_start(
          const dealii::VectorOperation::values operation,
          const unsigned int                    communication_channel,
          double *const                         data_this,
          const std::vector<double *> &         data_others,
          dealii::AlignedVector<double> &       buffer,
          std::vector<MPI_Request> &            requests) const override;

        /**
         * Finish to import from ghost array.
         */
        void
        import_from_ghosted_array_finish(
          const dealii::VectorOperation::values operation,
          double *const                         data_this,
          const std::vector<double *> &         data_others,
          const dealii::AlignedVector<double> & buffer,
          std::vector<MPI_Request> &            requests) const override;

        /**
         * Start to export to ghost array.
         */
        void
        export_to_ghosted_array_start(
          const unsigned int            communication_channel,
          float *const                  data_this,
          const std::vector<float *> &  data_others,
          dealii::AlignedVector<float> &buffer,
          std::vector<MPI_Request> &    requests) const override;

        /**
         * Finish to export to ghost array.
         */
        void
        export_to_ghosted_array_finish(
          float *const                data_this,
          const std::vector<float *> &data_others,
          std::vector<MPI_Request> &  requests) const override;

        /**
         * Start to import from ghost array.
         */
        void
        import_from_ghosted_array_start(
          const dealii::VectorOperation::values operation,
          const unsigned int                    communication_channel,
          float *const                          data_this,
          const std::vector<float *> &          data_others,
          dealii::AlignedVector<float> &        buffer,
          std::vector<MPI_Request> &            requests) const override;

        /**
         * Finish to import from ghost array.
         */
        void
        import_from_ghosted_array_finish(
          const dealii::VectorOperation::values operation,
          float *const                          data_this,
          const std::vector<float *> &          data_others,
          const dealii::AlignedVector<float> &  buffer,
          std::vector<MPI_Request> &            requests) const override;

        /**
         * TODO.
         */
        template <typename Number>
        void
        export_to_ghosted_array_finish_0(
          Number *const                data_this,
          const std::vector<Number *> &data_others,
          std::vector<MPI_Request> &   requests) const;

        /**
         * TODO.
         */
        template <typename Number>
        void
        export_to_ghosted_array_finish_1(
          Number *const                data_this,
          const std::vector<Number *> &data_others,
          std::vector<MPI_Request> &   requests) const;

      private:
        /**
         * Actual type-independent implementation of
         * export_to_ghosted_array_start().
         */
        template <typename Number>
        void
        export_to_ghosted_array_start_impl(
          const unsigned int             communication_channel,
          Number *const                  data_this,
          const std::vector<Number *> &  data_others,
          dealii::AlignedVector<Number> &buffer,
          std::vector<MPI_Request> &     requests) const;

        /**
         * Actual type-independent implementation of
         * export_to_ghosted_array_finish().
         */
        template <typename Number>
        void
        export_to_ghosted_array_finish_impl(
          Number *const                data_this,
          const std::vector<Number *> &data_others,
          std::vector<MPI_Request> &   requests) const;

        /**
         * Actual type-independent implementation of
         * import_from_ghosted_array_start().
         */
        template <typename Number>
        void
        import_from_ghosted_array_start_impl(
          const dealii::VectorOperation::values operation,
          const unsigned int                    communication_channel,
          Number *const                         data_this,
          const std::vector<Number *> &         data_others,
          dealii::AlignedVector<Number> &       buffer,
          std::vector<MPI_Request> &            requests) const;

        /**
         * Actual type-independent implementation of
         * import_from_ghosted_array_finish().
         */
        template <typename Number>
        void
        import_from_ghosted_array_finish_impl(
          const dealii::VectorOperation::values operation,
          Number *const                         data_this,
          const std::vector<Number *> &         data_others,
          const dealii::AlignedVector<Number> & buffer,
          std::vector<MPI_Request> &            requests) const;

      public:
        /**
         * Return position of shared cell: cell -> (owner, offset)
         */
        const std::map<dealii::types::global_dof_index,
                       std::pair<unsigned int, unsigned int>> &
        get_maps() const;


        /**
         * Return position of ghost face: (cell, no) -> (owner, offset)
         */
        const std::map<std::pair<dealii::types::global_dof_index, unsigned int>,
                       std::pair<unsigned int, unsigned int>> &
        get_maps_ghost() const;

        /**
         * Return memory consumption.
         *
         * @note: Only counts the buffers [TODO].
         */
        std::size_t
        memory_consumption() const;

        /**
         * Synchronize.
         */
        void
        sync() const;

      private:
        // I) configuration parameters
        bool         do_buffering;   // buffering vs. non-buffering modus
        unsigned int dofs_per_ghost; // ghost face or ghost cell

        // II) MPI-communicator related stuff
        unsigned int sm_size;
        unsigned int sm_rank;

        // III) access cells and ghost faces
        std::map<CellIdType, std::pair<RankType, LocalDoFType>> maps;
        std::map<FaceIdType, std::pair<RankType, LocalDoFType>> maps_ghost;

        // III) information to pack/unpack buffers
        std::vector<unsigned int>                    send_ranks;
        std::vector<dealii::types::global_dof_index> send_ptr;
        std::vector<dealii::types::global_dof_index> send_data_id;
        std::vector<unsigned int>                    send_data_face_no;

        std::vector<unsigned int>                    recv_ranks;
        std::vector<dealii::types::global_dof_index> recv_ptr;
        std::vector<dealii::types::global_dof_index> recv_size;

        std::vector<unsigned int> sm_targets;
        std::vector<unsigned int> sm_sources;


        std::vector<dealii::types::global_dof_index> sm_send_ptr;
        std::vector<unsigned int>                    sm_send_rank;
        std::vector<unsigned int>                    sm_send_offset_1;
        std::vector<unsigned int>                    sm_send_offset_2;
        std::vector<unsigned int>                    sm_send_no;

        std::vector<dealii::types::global_dof_index> sm_recv_ptr;
        std::vector<unsigned int>                    sm_recv_rank;
        std::vector<unsigned int>                    sm_recv_offset_1;
        std::vector<unsigned int>                    sm_recv_offset_2;
        std::vector<unsigned int>                    sm_recv_no;
      };


      void
      Partitioner::export_to_ghosted_array_start(
        const unsigned int             communication_channel,
        double *const                  data_this,
        const std::vector<double *> &  data_others,
        dealii::AlignedVector<double> &buffer,
        std::vector<MPI_Request> &     requests) const
      {
        this->export_to_ghosted_array_start_impl(
          communication_channel, data_this, data_others, buffer, requests);
      }



      void
      Partitioner::export_to_ghosted_array_finish(
        double *const                data_this,
        const std::vector<double *> &data_others,
        std::vector<MPI_Request> &   requests) const
      {
        this->export_to_ghosted_array_finish_impl(data_this,
                                                  data_others,
                                                  requests);
      }



      void
      Partitioner::import_from_ghosted_array_start(
        const dealii::VectorOperation::values operation,
        const unsigned int                    communication_channel,
        double *const                         data_this,
        const std::vector<double *> &         data_others,
        dealii::AlignedVector<double> &       buffer,
        std::vector<MPI_Request> &            requests) const
      {
        this->import_from_ghosted_array_start_impl(operation,
                                                   communication_channel,
                                                   data_this,
                                                   data_others,
                                                   buffer,
                                                   requests);
      }



      void
      Partitioner::import_from_ghosted_array_finish(
        const dealii::VectorOperation::values operation,
        double *const                         data_this,
        const std::vector<double *> &         data_others,
        const dealii::AlignedVector<double> & buffer,
        std::vector<MPI_Request> &            requests) const
      {
        this->import_from_ghosted_array_finish_impl(
          operation, data_this, data_others, buffer, requests);
      }



      void
      Partitioner::export_to_ghosted_array_start(
        const unsigned int            communication_channel,
        float *const                  data_this,
        const std::vector<float *> &  data_others,
        dealii::AlignedVector<float> &buffer,
        std::vector<MPI_Request> &    requests) const
      {
        export_to_ghosted_array_start_impl(
          communication_channel, data_this, data_others, buffer, requests);
      }



      void
      Partitioner::export_to_ghosted_array_finish(
        float *const                data_this,
        const std::vector<float *> &data_others,
        std::vector<MPI_Request> &  requests) const
      {
        this->export_to_ghosted_array_finish_impl(data_this,
                                                  data_others,
                                                  requests);
      }



      void
      Partitioner::import_from_ghosted_array_start(
        const dealii::VectorOperation::values operation,
        const unsigned int                    communication_channel,
        float *const                          data_this,
        const std::vector<float *> &          data_others,
        dealii::AlignedVector<float> &        buffer,
        std::vector<MPI_Request> &            requests) const
      {
        this->import_from_ghosted_array_start_impl(operation,
                                                   communication_channel,
                                                   data_this,
                                                   data_others,
                                                   buffer,
                                                   requests);
      }



      void
      Partitioner::import_from_ghosted_array_finish(
        const dealii::VectorOperation::values operation,
        float *const                          data_this,
        const std::vector<float *> &          data_others,
        const dealii::AlignedVector<float> &  buffer,
        std::vector<MPI_Request> &            requests) const
      {
        this->import_from_ghosted_array_finish_impl(
          operation, data_this, data_others, buffer, requests);
      }



      void
      Partitioner::sync() const
      {
        std::vector<MPI_Request> req(sm_targets.size() + sm_sources.size());

        for (unsigned int i = 0; i < sm_targets.size(); i++)
          {
            int dummy;
            MPI_Isend(&dummy,
                      0,
                      MPI_INT,
                      sm_targets[i],
                      mpi::internal::Tags::partitioner_sync,
                      this->comm_sm,
                      req.data() + i);
          }

        for (unsigned int i = 0; i < sm_sources.size(); i++)
          {
            int dummy;
            MPI_Irecv(&dummy,
                      0,
                      MPI_INT,
                      sm_sources[i],
                      mpi::internal::Tags::partitioner_sync,
                      this->comm_sm,
                      req.data() + i + sm_targets.size());
          }

        MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);
      }



      namespace internal
      {
        template <typename T, typename U>
        std::vector<std::pair<T, U>>
        MPI_Allgather_Pairs(const std::vector<std::pair<T, U>> &src,
                            const MPI_Comm &                    comm)
        {
          int size;
          MPI_Comm_size(comm, &size);

          std::vector<T> src_1;
          std::vector<U> src_2;

          for (auto i : src)
            {
              src_1.push_back(i.first);
              src_2.push_back(i.second);
            }


          unsigned int     len_local = src_1.size();
          std::vector<int> len_global(
            size); // actually unsigned int but MPI wants int
          MPI_Allgather(&len_local,
                        1,
                        MPI_INT,
                        &len_global[0],
                        1,
                        dealii::Utilities::MPI::internal::mpi_type_id(
                          &len_local),
                        comm);


          std::vector<int> displs; // actually unsigned int but MPI wants int
          displs.push_back(0);

          int total_size = 0;

          for (auto i : len_global)
            {
              displs.push_back(i + displs.back());
              total_size += i;
            }

          std::vector<T> dst_1(total_size);
          std::vector<U> dst_2(total_size);
          MPI_Allgatherv(
            &src_1[0],
            len_local,
            dealii::Utilities::MPI::internal::mpi_type_id(&src_1[0]),
            &dst_1[0],
            &len_global[0],
            &displs[0],
            dealii::Utilities::MPI::internal::mpi_type_id(&dst_1[0]),
            comm);
          MPI_Allgatherv(
            &src_2[0],
            len_local,
            dealii::Utilities::MPI::internal::mpi_type_id(&src_2[0]),
            &dst_2[0],
            &len_global[0],
            &displs[0],
            dealii::Utilities::MPI::internal::mpi_type_id(&dst_2[0]),
            comm);

          std::vector<std::pair<T, U>> dst(total_size);

          for (unsigned int i = 0; i < dst_1.size(); i++)
            dst[i] = {dst_1[i], dst_2[i]};

          return dst;
        }
      } // namespace internal



      template <typename Number>
      Partitioner::Partitioner(const ShapeInfo<Number> &shape_info)
        : dealii::LinearAlgebra::SharedMPI::PartitionerBase(false)
        , dofs_per_cell(shape_info.dofs_per_cell)
        , dofs_per_face(shape_info.dofs_per_face)
        , face_to_cell_index_nodal(shape_info.face_to_cell_index_nodal)
      {}


      void
      Partitioner::reinit(const dealii::IndexSet &is_locally_owned,
                          const dealii::IndexSet &is_locally_ghost,
                          const MPI_Comm &        communicator)
      {
        AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());

        (void)is_locally_owned;
        (void)is_locally_ghost;
        (void)communicator;
      }



      void
      Partitioner::reinit(
        const std::vector<dealii::types::global_dof_index> local_cells,
        const std::vector<
          std::pair<dealii::types::global_dof_index, std::vector<unsigned int>>>
                       local_ghost_faces,
        const MPI_Comm comm,
        const MPI_Comm comm_sm,
        const bool     do_buffering)
      {
        // fill some information needed by PartitionerBase
        this->comm             = comm;
        this->comm_sm          = comm_sm;
        this->n_mpi_processes_ = dealii::Utilities::MPI::n_mpi_processes(comm);

        this->do_buffering = do_buffering;

        AssertThrow(local_cells.size() > 0,
                    dealii::ExcMessage("No local cells!"));

        this->n_local_elements = local_cells.size() * dofs_per_cell;

        // 1) determine if ghost faces or ghost cells are needed
        const dealii::types::global_dof_index dofs_per_ghost = [&]() {
          unsigned int result = dofs_per_face;

          for (const auto &ghost_faces : local_ghost_faces)
            for (const auto ghost_face : ghost_faces.second)
              if (ghost_face == dealii::numbers::invalid_unsigned_int)
                result = dofs_per_cell;
          return dealii::Utilities::MPI::max(result, comm);
        }();

        this->dofs_per_ghost = dofs_per_ghost;

        // const auto this->comm_sm  = create_sm(comm);
        const auto sm_procs = hyperdeal::mpi::procs_of_sm(comm, this->comm_sm);
        const auto sm_rank  = [&]() {
          const auto ptr =
            std::find(sm_procs.begin(),
                      sm_procs.end(),
                      dealii::Utilities::MPI::this_mpi_process(comm));

          AssertThrow(ptr != sm_procs.end(),
                      dealii::ExcMessage("Proc not found!"));

          return std::distance(sm_procs.begin(), ptr);
        }();

        this->sm_rank = sm_rank;
        this->sm_size = sm_procs.size();


        for (unsigned int i = 0; i < local_cells.size(); i++)
          this->maps[local_cells[i]] = {sm_rank, i * dofs_per_cell};


        // 2) determine which ghost face is shared or remote
        std::vector<
          std::pair<dealii::types::global_dof_index, std::vector<unsigned int>>>
          local_ghost_faces_remote, local_ghost_faces_shared;
        {
          const auto n_total_cells = dealii::Utilities::MPI::sum(
            static_cast<dealii::types::global_dof_index>(local_cells.size()),
            comm);

          dealii::IndexSet is_local_cells(n_total_cells);
          is_local_cells.add_indices(local_cells.begin(), local_cells.end());

          dealii::IndexSet is_ghost_cells(n_total_cells);
          for (const auto &ghost_faces : local_ghost_faces)
            is_ghost_cells.add_index(ghost_faces.first);

          for (unsigned int i = 0; i < local_ghost_faces.size(); i++)
            AssertThrow(local_ghost_faces[i].first ==
                          is_ghost_cells.nth_index_in_set(i),
                        dealii::ExcMessage("PROBLEM!"));

          AssertThrow(
            local_ghost_faces.size() == is_ghost_cells.n_elements(),
            dealii::ExcMessage(
              "Dimensions " + std::to_string(local_ghost_faces.size()) + " " +
              std::to_string(is_ghost_cells.n_elements()) + " do not match!"));

          std::vector<unsigned int> owning_ranks_of_ghosts(
            is_ghost_cells.n_elements());

          // set up dictionary
          dealii::Utilities::MPI::internal::ComputeIndexOwner::
            ConsensusAlgorithmsPayload process(is_local_cells,
                                               is_ghost_cells,
                                               comm,
                                               owning_ranks_of_ghosts,
                                               false);

          dealii::Utilities::MPI::ConsensusAlgorithms::Selector<
            std::pair<dealii::types::global_dof_index,
                      dealii::types::global_dof_index>,
            unsigned int>
            consensus_algorithm(process, comm);
          consensus_algorithm.run();

          std::map<unsigned int, std::vector<dealii::types::global_dof_index>>
            shared_procs_to_cells;

          for (unsigned int i = 0; i < owning_ranks_of_ghosts.size(); i++)
            {
              AssertThrow(dealii::Utilities::MPI::this_mpi_process(comm) !=
                            owning_ranks_of_ghosts[i],
                          dealii::ExcMessage(
                            "Locally owned cells should be not ghosted!"));

              const auto ptr = std::find(sm_procs.begin(),
                                         sm_procs.end(),
                                         owning_ranks_of_ghosts[i]);

              if (ptr == sm_procs.end())
                local_ghost_faces_remote.push_back(local_ghost_faces[i]);
              else
                {
                  local_ghost_faces_shared.push_back(local_ghost_faces[i]);
                  shared_procs_to_cells[std::distance(sm_procs.begin(), ptr)]
                    .emplace_back(local_ghost_faces[i].first);
                }
            }

          // determine (ghost sm cell) -> (sm rank, offset)
          dealii::Utilities::MPI::ConsensusAlgorithms::
            AnonymousProcess<dealii::types::global_dof_index, unsigned int>
              temp(
                [&]() {
                  std::vector<unsigned int> result;
                  for (auto &i : shared_procs_to_cells)
                    result.push_back(i.first);
                  return result;
                },
                [&](const auto other_rank, auto &send_buffer) {
                  send_buffer = shared_procs_to_cells[other_rank];
                },
                [&](const auto &other_rank,
                    const auto &buffer_recv,
                    auto &      request_buffer) {
                  request_buffer.resize(buffer_recv.size());

                  for (unsigned int i = 0; i < buffer_recv.size(); i++)
                    {
                      const auto value = buffer_recv[i];
                      const auto ptr   = std::find(local_cells.begin(),
                                                 local_cells.end(),
                                                 value);

                      AssertThrow(
                        ptr != local_cells.end(),
                        dealii::ExcMessage(
                          "Cell " + std::to_string(value) + " at index " +
                          std::to_string(i) + " on rank " +
                          std::to_string(
                            dealii::Utilities::MPI::this_mpi_process(comm)) +
                          " requested by rank " + std::to_string(other_rank) +
                          " not found!"));

                      request_buffer[i] =
                        std::distance(local_cells.begin(), ptr);
                    }
                },
                [&](const auto other_rank, auto &recv_buffer) {
                  recv_buffer.resize(shared_procs_to_cells[other_rank].size());
                },
                [&](const auto other_rank, const auto &recv_buffer) {
                  for (unsigned int i = 0; i < recv_buffer.size(); i++)
                    {
                      const dealii::types::global_dof_index cell =
                        shared_procs_to_cells[other_rank][i];
                      const unsigned int offset = recv_buffer[i];

                      Assert(maps.find(cell) == maps.end(),
                             dealii::ExcMessage("Cell " + std::to_string(cell) +
                                                " is already in maps!"));

                      this->maps[cell] = {other_rank, offset * dofs_per_cell};
                    }
                });
          dealii::Utilities::MPI::ConsensusAlgorithms::Selector<
            dealii::types::global_dof_index,
            unsigned int>(temp, this->comm_sm)
            .run();
        }


        // 3) merge local_ghost_faces_remote and sort -> ghost_faces_remote
        const auto local_ghost_faces_remote_pairs_global =
          [&local_ghost_faces_remote, &comm, this]() {
            std::vector<
              std::pair<dealii::types::global_dof_index, unsigned int>>
              local_ghost_faces_remote_pairs_local;

            // convert vector<pair<U, std::vector<V>>> ->.vector<std::pair<U,
            // V>>>
            for (const auto &ghost_faces : local_ghost_faces_remote)
              for (const auto ghost_face : ghost_faces.second)
                local_ghost_faces_remote_pairs_local.emplace_back(
                  ghost_faces.first, ghost_face);

            // collect all on which are shared
            std::vector<
              std::pair<dealii::types::global_dof_index, unsigned int>>
              local_ghost_faces_remote_pairs_global =
                internal::MPI_Allgather_Pairs(
                  local_ghost_faces_remote_pairs_local, this->comm_sm);

            // sort
            std::sort(local_ghost_faces_remote_pairs_global.begin(),
                      local_ghost_faces_remote_pairs_global.end());

            return local_ghost_faces_remote_pairs_global;
          }();


        // 4) distributed ghost_faces_remote,
        auto distributed_local_ghost_faces_remote_pairs_global =
          [&local_ghost_faces_remote_pairs_global, &sm_procs]() {
            std::vector<std::vector<
              std::pair<dealii::types::global_dof_index, unsigned int>>>
              result(sm_procs.size());

            unsigned int       counter = 0;
            const unsigned int faces_per_process =
              (local_ghost_faces_remote_pairs_global.size() + sm_procs.size() -
               1) /
              sm_procs.size();
            for (auto p : local_ghost_faces_remote_pairs_global)
              result[(counter++) / faces_per_process].push_back(p);

            return result;
          }();

        // revert partitioning of ghost faces (TODO)
        {
          std::vector<std::pair<dealii::types::global_dof_index, unsigned int>>
            local_ghost_faces_remote_pairs_local;

          for (const auto &ghost_faces : local_ghost_faces_remote)
            for (const auto ghost_face : ghost_faces.second)
              local_ghost_faces_remote_pairs_local.emplace_back(
                ghost_faces.first, ghost_face);

          distributed_local_ghost_faces_remote_pairs_global.clear();
          distributed_local_ghost_faces_remote_pairs_global.resize(
            sm_procs.size());
          distributed_local_ghost_faces_remote_pairs_global[sm_rank] =
            local_ghost_faces_remote_pairs_local;
        }


        // ... update ghost size, and
        this->n_ghost_elements =
          (distributed_local_ghost_faces_remote_pairs_global[sm_rank].size() +
           (do_buffering ?
              std::accumulate(local_ghost_faces_shared.begin(),
                              local_ghost_faces_shared.end(),
                              std::size_t(0),
                              [](const std::size_t &a, const auto &b) {
                                return a + b.second.size();
                              }) :
              std::size_t(0))) *
          dofs_per_ghost;


        // ... update ghost map
        this->maps_ghost = [&]() {
          std::map<std::pair<dealii::types::global_dof_index, unsigned int>,
                   std::pair<unsigned int, unsigned int>>
            maps_ghost;

          // counter for offset of ghost faces
          unsigned int my_offset = local_cells.size() * dofs_per_cell;

          // buffering-mode: insert shared faces into maps_ghost
          if (do_buffering)
            for (const auto &pair : local_ghost_faces_shared)
              for (const auto &face : pair.second)
                {
                  maps_ghost[{pair.first, face}] = {sm_rank, my_offset};
                  my_offset += dofs_per_ghost;
                }

          // create map (cell, face_no) -> (sm rank, offset)
          const auto maps_ghost_inverse =
            [&distributed_local_ghost_faces_remote_pairs_global,
             &dofs_per_ghost,
             &local_cells,
             this,
             &sm_procs,
             &my_offset]() {
              std::vector<unsigned int> offsets(sm_procs.size());

              MPI_Allgather(
                &my_offset,
                1,
                dealii::Utilities::MPI::internal::mpi_type_id(&my_offset),
                offsets.data(),
                1,
                dealii::Utilities::MPI::internal::mpi_type_id(&my_offset),
                this->comm_sm);

              std::map<std::pair<unsigned int, unsigned int>,
                       std::pair<dealii::types::global_dof_index, unsigned int>>
                maps_ghost_inverse;

              for (unsigned int i = 0; i < sm_procs.size(); i++)
                for (unsigned int j = 0;
                     j < distributed_local_ghost_faces_remote_pairs_global[i]
                           .size();
                     j++)
                  maps_ghost_inverse
                    [distributed_local_ghost_faces_remote_pairs_global[i][j]] =
                      {i, offsets[i] + j * dofs_per_ghost};

              return maps_ghost_inverse;
            }();

          // create map (cell, face_no) -> (sm rank, offset) with only
          // ghost faces needed for evaluation
          for (const auto &i : local_ghost_faces_remote)
            for (const auto &j : i.second)
              maps_ghost[{i.first, j}] = maps_ghost_inverse.at({i.first, j});

          for (const auto &i :
               distributed_local_ghost_faces_remote_pairs_global[sm_rank])
            maps_ghost[i] = maps_ghost_inverse.at(i);

          return maps_ghost;
        }();

        // buffering-mode: pre-compute information for memcopy of ghost faces
        std::vector<std::array<LocalDoFType, 5>> ghost_list_shared_precomp;
        std::vector<std::array<LocalDoFType, 5>> maps_ghost_inverse_precomp;

        if (do_buffering)
          {
            std::map<unsigned int, std::vector<std::array<LocalDoFType, 5>>>
              send_data;

            for (const auto &pair : local_ghost_faces_shared)
              for (const auto &face : pair.second)
                {
                  const auto ptr1 = maps.find(pair.first);
                  AssertThrow(ptr1 != maps.end(),
                              dealii::ExcMessage("Entry not found!"));

                  const auto ptr2 = maps_ghost.find({pair.first, face});
                  AssertThrow(ptr2 != maps_ghost.end(),
                              dealii::ExcMessage("Entry not found!"));

                  std::array<LocalDoFType, 5> v{{ptr1->second.first,
                                                 ptr1->second.second,
                                                 face,
                                                 ptr2->second.first,
                                                 ptr2->second.second}};

                  ghost_list_shared_precomp.push_back(v);

                  send_data[ptr1->second.first].push_back(v);
                }

            dealii::Utilities::MPI::ConsensusAlgorithms::
              AnonymousProcess<LocalDoFType, LocalDoFType>
                temp(
                  [&]() {
                    std::vector<unsigned int> result;
                    for (auto &i : send_data)
                      result.push_back(i.first);
                    return result;
                  },
                  [&](const auto other_rank, auto &send_buffer) {
                    for (const auto &i : send_data[other_rank])
                      for (const auto &j : i)
                        send_buffer.push_back(j);
                  },
                  [&](const auto & /*other_rank*/,
                      const auto &buffer_recv,
                      auto & /*request_buffer*/) {
                    for (unsigned int i = 0; i < buffer_recv.size(); i += 5)
                      maps_ghost_inverse_precomp.push_back(
                        {{buffer_recv[i],
                          buffer_recv[i + 1],
                          buffer_recv[i + 2],
                          buffer_recv[i + 3],
                          buffer_recv[i + 4]}});
                  });
            dealii::Utilities::MPI::ConsensusAlgorithms::Selector<LocalDoFType,
                                                                  LocalDoFType>(
              temp, this->comm_sm)
              .run();

            std::sort(maps_ghost_inverse_precomp.begin(),
                      maps_ghost_inverse_precomp.end());
          }

        // rank -> pair(offset, size)
        std::map<unsigned int, std::pair<unsigned int, unsigned int>>
          receive_info;
        std::map<unsigned int, std::vector<std::array<unsigned int, 3>>>
          requests_from_relevant_precomp;

        // 5) setup communication patterns (during update_ghost_values &
        // compress)
        [&local_cells,
         &distributed_local_ghost_faces_remote_pairs_global,
         &sm_rank,
         &comm,
         &dofs_per_ghost](auto &      requests_from_relevant_precomp,
                          auto &      receive_info,
                          const auto &maps,
                          const auto &maps_ghost) {
          // determine of the owner of cells of remote ghost faces
          const auto n_total_cells = dealii::Utilities::MPI::sum(
            static_cast<dealii::types::global_dof_index>(local_cells.size()),
            comm);

          // owned cells (TODO: generalize so that local_cells is also
          // partitioned)
          dealii::IndexSet is_local_cells(n_total_cells);
          is_local_cells.add_indices(local_cells.begin(), local_cells.end());

          // needed (ghost) cell
          dealii::IndexSet is_ghost_cells(n_total_cells);
          for (const auto &ghost_faces :
               distributed_local_ghost_faces_remote_pairs_global[sm_rank])
            is_ghost_cells.add_index(ghost_faces.first);

          // determine rank of (ghost) cells
          const auto owning_ranks_of_ghosts = [&]() {
            std::vector<unsigned int> owning_ranks_of_ghosts(
              is_ghost_cells.n_elements());

            dealii::Utilities::MPI::internal::ComputeIndexOwner::
              ConsensusAlgorithmsPayload process(is_local_cells,
                                                 is_ghost_cells,
                                                 comm,
                                                 owning_ranks_of_ghosts,
                                                 false);

            dealii::Utilities::MPI::ConsensusAlgorithms::Selector<
              std::pair<dealii::types::global_dof_index,
                        dealii::types::global_dof_index>,
              unsigned int>
              consensus_algorithm(process, comm);
            consensus_algorithm.run();

            return owning_ranks_of_ghosts;
          }();

          // determine targets
          const auto send_ranks = [&]() {
            std::set<unsigned int> send_ranks_set;

            for (const auto &i : owning_ranks_of_ghosts)
              send_ranks_set.insert(i);

            const std::vector<unsigned int> send_ranks(send_ranks_set.begin(),
                                                       send_ranks_set.end());

            return send_ranks;
          }();

          // collect ghost faces (separated for each target)
          const auto send_data = [&]() {
            std::vector<std::vector<std::pair<dealii::types::global_dof_index,
                                              dealii::types::global_dof_index>>>
              send_data(send_ranks.size());

            unsigned int index      = 0;
            unsigned int index_cell = dealii::numbers::invalid_unsigned_int;

            for (const auto &ghost_faces :
                 distributed_local_ghost_faces_remote_pairs_global[sm_rank])
              {
                if (index_cell != ghost_faces.first)
                  {
                    index_cell = ghost_faces.first;
                    const unsigned int index_rank =
                      owning_ranks_of_ghosts[is_ghost_cells.index_within_set(
                        ghost_faces.first)];
                    index = std::distance(send_ranks.begin(),
                                          std::find(send_ranks.begin(),
                                                    send_ranks.end(),
                                                    index_rank));
                  }
                send_data[index].emplace_back(ghost_faces.first,
                                              ghost_faces.second);
              }

            return send_data;
          }();

          // send ghost faces to the owners
          std::vector<MPI_Request> send_requests(send_ranks.size());

          for (unsigned int i = 0; i < send_ranks.size(); i++)
            {
              dealii::types::global_dof_index dummy;
              MPI_Isend(send_data[i].data(),
                        2 * send_data[i].size(),
                        dealii::Utilities::MPI::internal::mpi_type_id(&dummy),
                        send_ranks[i],
                        105,
                        comm,
                        send_requests.data() + i);

              receive_info[send_ranks[i]] = {
                send_data[i].size() * dofs_per_ghost,
                maps_ghost.at(send_data[i][0]).second};
            }

          MPI_Barrier(comm);

          const auto targets = dealii::Utilities::MPI::
            compute_point_to_point_communication_pattern(comm, send_ranks);

          // process requests
          for (unsigned int i = 0; i < targets.size(); i++)
            {
              // wait for any request
              MPI_Status status;
              auto       ierr = MPI_Probe(MPI_ANY_SOURCE, 105, comm, &status);
              AssertThrowMPI(ierr);

              // determine number of ghost faces * 2 (since we are considering
              // pairs)
              int                             len;
              dealii::types::global_dof_index dummy;
              MPI_Get_count(&status,
                            dealii::Utilities::MPI::internal::mpi_type_id(
                              &dummy),
                            &len);

              AssertThrow(len % 2 == 0,
                          dealii::ExcMessage("Length " + std::to_string(len) +
                                             " is not a multiple of two!"));

              // allocate memory for the incoming vector
              std::vector<std::pair<dealii::types::global_dof_index,
                                    dealii::types::global_dof_index>>
                recv_data(len / 2);

              // receive data
              ierr =
                MPI_Recv(recv_data.data(),
                         len,
                         dealii::Utilities::MPI::internal::mpi_type_id(&dummy),
                         status.MPI_SOURCE,
                         status.MPI_TAG,
                         comm,
                         &status);
              AssertThrowMPI(ierr);

              // setup pack and unpack info
              requests_from_relevant_precomp[status.MPI_SOURCE] = [&]() {
                std::vector<std::array<unsigned int, 3>> temp(len / 2);
                for (unsigned int i = 0; i < static_cast<unsigned int>(len) / 2;
                     i++)
                  {
                    const CellIdType   cell    = recv_data[i].first;
                    const unsigned int face_no = recv_data[i].second;

                    const auto ptr = maps.find(cell);
                    AssertThrow(ptr != maps.end(),
                                dealii::ExcMessage("Entry " +
                                                   std::to_string(cell) +
                                                   " not found!"));

                    temp[i] = std::array<unsigned int, 3>{
                      {ptr->second.first,
                       (unsigned int)ptr->second.second,
                       face_no}};
                  }
                return temp;
              }();
            }

          // make sure requests have been sent away
          MPI_Waitall(send_requests.size(),
                      send_requests.data(),
                      MPI_STATUSES_IGNORE);
        }(requests_from_relevant_precomp,
          receive_info,
          this->maps,
          this->maps_ghost);


        {
          recv_ptr.clear();
          recv_size.clear();
          recv_ranks.clear();

          for (const auto i : receive_info)
            {
              recv_ranks.push_back(i.first);
              recv_size.push_back(i.second.first);
              recv_ptr.push_back(i.second.second);
            }
        }

        {
          // TODO: clear
          send_ptr.push_back(0);

          for (const auto &i : requests_from_relevant_precomp)
            {
              send_ranks.push_back(i.first);

              for (const auto &j : i.second)
                {
                  AssertThrow(j[0] == sm_rank,
                              dealii::StandardExceptions::ExcNotImplemented());
                  send_data_id.push_back(j[1]);
                  send_data_face_no.push_back(j[2]);
                }

              send_ptr.push_back(send_data_id.size());
            }
        }

        {
          std::set<unsigned int> temp;

          for (const auto &i : maps)
            if (i.second.first != sm_rank)
              temp.insert(i.second.first);

          for (const auto &i : temp)
            this->sm_sources.push_back(i);

          this->sm_targets = dealii::Utilities::MPI::
            compute_point_to_point_communication_pattern(this->comm_sm,
                                                         sm_sources);
        }

        if (do_buffering)
          {
            auto temp = ghost_list_shared_precomp;
            std::sort(temp.begin(), temp.end());

            std::vector<
              std::vector<std::array<dealii::types::global_dof_index, 3>>>
              temp_(sm_size);

            for (auto t : temp)
              {
                AssertThrow(sm_rank == t[3],
                            dealii::StandardExceptions::ExcNotImplemented());
                temp_[t[0]].emplace_back(
                  std::array<dealii::types::global_dof_index, 3>{
                    {t[1], t[2], t[4]}});
              }


            sm_send_ptr.push_back(0);

            for (unsigned int i = 0; i < temp_.size(); i++)
              {
                if (temp_[i].size() == 0)
                  continue;

                sm_send_rank.push_back(i);

                for (const auto &v : temp_[i])
                  {
                    sm_send_offset_1.push_back(v[2]);
                    sm_send_offset_2.push_back(v[0]);
                    sm_send_no.push_back(v[1]);
                  }

                sm_send_ptr.push_back(sm_send_no.size());
              }

            AssertThrow(sm_send_rank.size() == sm_sources.size(),
                        dealii::StandardExceptions::ExcNotImplemented());
          }

        if (do_buffering)
          {
            auto temp = maps_ghost_inverse_precomp;
            std::sort(temp.begin(), temp.end());

            std::vector<
              std::vector<std::array<dealii::types::global_dof_index, 3>>>
              temp_(sm_size);

            for (auto t : temp)
              {
                AssertThrow(sm_rank == t[0],
                            dealii::StandardExceptions::ExcNotImplemented());
                temp_[t[3]].emplace_back(
                  std::array<dealii::types::global_dof_index, 3>{
                    {t[4], t[2], t[1]}});
              }


            sm_recv_ptr.push_back(0);

            for (unsigned int i = 0; i < temp_.size(); i++)
              {
                if (temp_[i].size() == 0)
                  continue;

                sm_recv_rank.push_back(i);

                for (const auto &v : temp_[i])
                  {
                    sm_recv_offset_1.push_back(v[2]);
                    sm_recv_offset_2.push_back(v[0]);
                    sm_recv_no.push_back(v[1]);
                  }

                sm_recv_ptr.push_back(sm_recv_no.size());
              }

            AssertThrow(sm_recv_rank.size() == sm_targets.size(),
                        dealii::StandardExceptions::ExcNotImplemented());
          }
      }



      template <typename Number>
      void
      Partitioner::export_to_ghosted_array_start_impl(
        const unsigned int communication_channel,
        Number *const      data_this,
        const std::vector<Number *> & /*data_others*/,
        dealii::AlignedVector<Number> &send_buffer_data,
        std::vector<MPI_Request> &     requests) const
      {
        if (send_buffer_data.size() == 0)
          {
            send_buffer_data.resize_fast(send_ptr.back() * dofs_per_ghost);
          }
        else
          {
            AssertThrow(send_buffer_data.size() ==
                          send_ptr.back() * dofs_per_ghost,
                        dealii::StandardExceptions::ExcDimensionMismatch(
                          send_buffer_data.size(),
                          send_ptr.back() * dofs_per_ghost));
          }

        requests.resize(sm_sources.size() + sm_targets.size() +
                        recv_ranks.size() + send_ranks.size());

        // 1) notify relevant shared processes that local data is available
        if (sm_size > 1)
          {
            int dummy;
            for (unsigned int i = 0; i < sm_targets.size(); i++)
              MPI_Isend(&dummy,
                        0,
                        MPI_INT,
                        sm_targets[i],
                        communication_channel + 21,
                        this->comm_sm,
                        requests.data() + i + sm_sources.size());

            for (unsigned int i = 0; i < sm_sources.size(); i++)
              MPI_Irecv(&dummy,
                        0,
                        MPI_INT,
                        sm_sources[i],
                        communication_channel + 21,
                        this->comm_sm,
                        requests.data() + i);
          }

        // 2) start receiving form (remote) processes
        {
          for (unsigned int i = 0; i < recv_ranks.size(); i++)
            MPI_Irecv(data_this + recv_ptr[i],
                      recv_size[i],
                      MPI_DOUBLE,
                      recv_ranks[i],
                      communication_channel + 22,
                      comm,
                      requests.data() + i + sm_sources.size() +
                        sm_targets.size());
        }

        // 3) fill buffers and start sending to (remote) processes
        for (unsigned int c = 0; c < send_ranks.size(); c++)
          {
            auto buffer =
              send_buffer_data.data() + send_ptr[c] * dofs_per_ghost;

            for (unsigned int i = send_ptr[c]; i < send_ptr[c + 1];
                 i++, buffer += dofs_per_ghost)
              if (dofs_per_ghost == dofs_per_face)
                {
                  auto *__restrict dst       = buffer;
                  const auto *__restrict src = data_this + send_data_id[i];
                  const auto *__restrict idx =
                    face_to_cell_index_nodal[send_data_face_no[i]].data();

                  for (unsigned int i = 0; i < dofs_per_face; i++)
                    dst[i] = src[idx[i]];
                }
              else if (dofs_per_ghost == dofs_per_cell)
                {
                  AssertThrow(false,
                              dealii::StandardExceptions::ExcNotImplemented());
                }

            MPI_Issend(send_buffer_data.data() + send_ptr[c] * dofs_per_ghost,
                       (send_ptr[c + 1] - send_ptr[c]) * dofs_per_ghost,
                       MPI_DOUBLE,
                       send_ranks[c],
                       communication_channel + 22,
                       comm,
                       requests.data() + c + sm_sources.size() +
                         sm_targets.size() + recv_ranks.size());
          }
      }



      template <typename Number>
      void
      Partitioner::export_to_ghosted_array_finish_impl(
        Number *const                data_this,
        const std::vector<Number *> &data_others,
        std::vector<MPI_Request> &   requests) const
      {
        AssertDimension(requests.size(),
                        sm_sources.size() + sm_targets.size() +
                          recv_ranks.size() + send_ranks.size());

        if (do_buffering) // deal with shared faces if buffering is requested
          {
            // update ghost values of shared cells (if requested)
            for (unsigned int c = 0; c < sm_sources.size(); c++)
              {
                int        i;
                MPI_Status status;
                const auto ierr =
                  MPI_Waitany(sm_sources.size(), requests.data(), &i, &status);
                AssertThrowMPI(ierr);

                for (unsigned int j = sm_send_ptr[i]; j < sm_send_ptr[i + 1];
                     j++)
                  if (dofs_per_ghost == dofs_per_face)
                    {
                      auto *__restrict dst = data_this + sm_send_offset_1[j];
                      const auto *__restrict src =
                        data_others[sm_send_rank[i]] + sm_send_offset_2[j];
                      const auto *__restrict idx =
                        face_to_cell_index_nodal[sm_send_no[j]].data();

                      for (unsigned int i = 0; i < dofs_per_face; i++)
                        dst[i] = src[idx[i]];
                    }
                  else if (dofs_per_ghost == dofs_per_cell)
                    {
                      AssertThrow(
                        false, dealii::StandardExceptions::ExcNotImplemented());
                    }
              }
          }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      }



      template <typename Number>
      void
      Partitioner::export_to_ghosted_array_finish_0(
        Number *const                data_this,
        const std::vector<Number *> &data_others,
        std::vector<MPI_Request> &   requests) const
      {
        AssertDimension(requests.size(),
                        sm_sources.size() + sm_targets.size() +
                          recv_ranks.size() + send_ranks.size());

        if (do_buffering) // deal with shared faces if buffering is requested
          {
            // update ghost values of shared cells (if requested)
            for (unsigned int c = 0; c < sm_sources.size(); c++)
              {
                int        i;
                MPI_Status status;
                const auto ierr =
                  MPI_Waitany(sm_sources.size(), requests.data(), &i, &status);
                AssertThrowMPI(ierr);

                for (unsigned int j = sm_send_ptr[i]; j < sm_send_ptr[i + 1];
                     j++)
                  if (dofs_per_ghost == dofs_per_face)
                    {
                      auto *__restrict dst = data_this + sm_send_offset_1[j];
                      const auto *__restrict src =
                        data_others[sm_send_rank[i]] + sm_send_offset_2[j];
                      const auto *__restrict idx =
                        face_to_cell_index_nodal[sm_send_no[j]].data();

                      for (unsigned int i = 0; i < dofs_per_face; i++)
                        dst[i] = src[idx[i]];
                    }
                  else if (dofs_per_ghost == dofs_per_cell)
                    {
                      AssertThrow(
                        false, dealii::StandardExceptions::ExcNotImplemented());
                    }
              }
          }
        else
          {
            MPI_Waitall(sm_sources.size(),
                        requests.data(),
                        MPI_STATUSES_IGNORE);
          }
      }



      template <typename Number>
      void
      Partitioner::export_to_ghosted_array_finish_1(
        Number *const                data_this,
        const std::vector<Number *> &data_others,
        std::vector<MPI_Request> &   requests) const
      {
        (void)data_this;
        (void)data_others;

        AssertDimension(requests.size(),
                        sm_sources.size() + sm_targets.size() +
                          recv_ranks.size() + send_ranks.size());

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      }



      template <typename Number>
      void
      Partitioner::import_from_ghosted_array_start_impl(
        const dealii::VectorOperation::values operation,
        const unsigned int                    communication_channel,
        Number *const                         data_this,
        const std::vector<Number *> &         data_others,
        dealii::AlignedVector<Number> &       send_buffer_data,
        std::vector<MPI_Request> &            requests) const
      {
        (void)data_others;
        (void)communication_channel;

        AssertThrow(operation == dealii::VectorOperation::add,
                    dealii::ExcMessage("Not yet implemented."));

        if (send_buffer_data.size() == 0)
          {
            send_buffer_data.resize_fast(send_ptr.back() * dofs_per_ghost);
          }
        else
          {
            AssertThrow(send_buffer_data.size() ==
                          send_ptr.back() * dofs_per_ghost,
                        dealii::StandardExceptions::ExcDimensionMismatch(
                          send_buffer_data.size(),
                          send_ptr.back() * dofs_per_ghost));
          }

        requests.resize(sm_sources.size() + sm_targets.size() +
                        recv_ranks.size() + send_ranks.size());

        // 1) notify relevant shared processes that data is available
        if (sm_size > 1)
          {
            int dummy;

            for (unsigned int i = 0; i < sm_sources.size(); i++)
              MPI_Isend(&dummy,
                        0,
                        MPI_INT,
                        sm_sources[i],
                        communication_channel + 21,
                        this->comm_sm,
                        requests.data() + i);

            for (unsigned int i = 0; i < sm_targets.size(); i++)
              MPI_Irecv(&dummy,
                        0,
                        MPI_INT,
                        sm_targets[i],
                        communication_channel + 21,
                        this->comm_sm,
                        requests.data() + i + sm_sources.size());
          }

        // request receive
        {
          for (unsigned int i = 0; i < recv_ranks.size(); i++)
            MPI_Isend(data_this + recv_ptr[i],
                      recv_size[i],
                      MPI_DOUBLE,
                      recv_ranks[i],
                      0,
                      comm,
                      requests.data() + i + sm_sources.size() +
                        sm_targets.size());
        }

        // fill buffers and request send
        for (unsigned int i = 0; i < send_ranks.size(); i++)
          MPI_Irecv(send_buffer_data.data() + send_ptr[i] * dofs_per_ghost,
                    (send_ptr[i + 1] - send_ptr[i]) * dofs_per_ghost,
                    MPI_DOUBLE,
                    send_ranks[i],
                    0,
                    comm,
                    requests.data() + i + sm_sources.size() +
                      sm_targets.size() + recv_ranks.size());
      }



      template <typename Number>
      void
      Partitioner::import_from_ghosted_array_finish_impl(
        const dealii::VectorOperation::values operation,
        Number *const                         data_this,
        const std::vector<Number *> &         data_others,
        const dealii::AlignedVector<Number> & send_buffer_data,
        std::vector<MPI_Request> &            requests) const
      {
        AssertThrow(operation == dealii::VectorOperation::add,
                    dealii::ExcMessage("Not yet implemented."));

        AssertThrow(send_buffer_data.size() == send_ptr.back() * dofs_per_ghost,
                    dealii::StandardExceptions::ExcDimensionMismatch(
                      send_buffer_data.size(),
                      send_ptr.back() * dofs_per_ghost));

        AssertDimension(requests.size(),
                        sm_sources.size() + sm_targets.size() +
                          recv_ranks.size() + send_ranks.size());

        // 1) compress for shared faces
        if (do_buffering)
          {
            for (unsigned int c = 0; c < sm_targets.size(); c++)
              {
                int        i;
                MPI_Status status;
                const auto ierr =
                  MPI_Waitany(sm_targets.size(),
                              requests.data() + sm_sources.size(),
                              &i,
                              &status);
                AssertThrowMPI(ierr);

                for (unsigned int j = sm_recv_ptr[i]; j < sm_recv_ptr[i + 1];
                     j++)
                  if (dofs_per_ghost == dofs_per_face)
                    {
                      auto *__restrict dst = data_this + sm_recv_offset_1[j];
                      const auto *__restrict src =
                        data_others[sm_recv_rank[i]] + sm_recv_offset_2[j];
                      const auto *__restrict idx =
                        face_to_cell_index_nodal[sm_recv_no[j]].data();

                      for (unsigned int i = 0; i < dofs_per_face; i++)
                        dst[idx[i]] += src[i];
                    }
                  else if (dofs_per_ghost == dofs_per_cell)
                    {
                      AssertThrow(
                        false, dealii::StandardExceptions::ExcNotImplemented());
                    }
              }
          }

        // 2) receive data and compress for remote faces
        for (unsigned int c = 0; c < send_ranks.size(); c++)
          {
            int        r;
            MPI_Status status;
            const auto ierr =
              MPI_Waitany(send_ranks.size(),
                          requests.data() + sm_sources.size() +
                            sm_targets.size() + recv_ranks.size(),
                          &r,
                          &status);
            AssertThrowMPI(ierr);

            auto buffer =
              send_buffer_data.data() + send_ptr[r] * dofs_per_ghost;

            for (unsigned int i = send_ptr[r]; i < send_ptr[r + 1];
                 i++, buffer += dofs_per_ghost)
              if (dofs_per_ghost == dofs_per_face)
                {
                  auto *__restrict dst       = data_this + send_data_id[i];
                  const auto *__restrict src = buffer;
                  const auto *__restrict idx =
                    face_to_cell_index_nodal[send_data_face_no[i]].data();

                  for (unsigned int i = 0; i < dofs_per_face; i++)
                    dst[idx[i]] += src[i];
                }
              else if (dofs_per_ghost == dofs_per_cell)
                {
                  AssertThrow(false,
                              dealii::StandardExceptions::ExcNotImplemented());
                }
          }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      }



      const std::map<dealii::types::global_dof_index,
                     std::pair<unsigned int, unsigned int>> &
      Partitioner::get_maps() const
      {
        return maps;
      }



      const std::map<std::pair<dealii::types::global_dof_index, unsigned int>,
                     std::pair<unsigned int, unsigned int>> &
      Partitioner::get_maps_ghost() const
      {
        return maps_ghost;
      }



      std::size_t
      Partitioner::memory_consumption() const
      {
        // [TODO] not counting maps and maps_ghost

        return dealii::MemoryConsumption::memory_consumption(send_ranks) +
               dealii::MemoryConsumption::memory_consumption(send_ptr) +
               dealii::MemoryConsumption::memory_consumption(send_data_id) +
               dealii::MemoryConsumption::memory_consumption(
                 send_data_face_no) +
               dealii::MemoryConsumption::memory_consumption(recv_ranks) +
               dealii::MemoryConsumption::memory_consumption(recv_ptr) +
               dealii::MemoryConsumption::memory_consumption(recv_size) +
               dealii::MemoryConsumption::memory_consumption(sm_targets) +
               dealii::MemoryConsumption::memory_consumption(sm_sources) +
               dealii::MemoryConsumption::memory_consumption(sm_send_ptr) +
               dealii::MemoryConsumption::memory_consumption(sm_send_rank) +
               dealii::MemoryConsumption::memory_consumption(sm_send_offset_1) +
               dealii::MemoryConsumption::memory_consumption(sm_send_offset_2) +
               dealii::MemoryConsumption::memory_consumption(sm_send_no) +
               dealii::MemoryConsumption::memory_consumption(sm_recv_ptr) +
               dealii::MemoryConsumption::memory_consumption(sm_recv_rank) +
               dealii::MemoryConsumption::memory_consumption(sm_recv_offset_1) +
               dealii::MemoryConsumption::memory_consumption(sm_recv_offset_2) +
               dealii::MemoryConsumption::memory_consumption(sm_recv_no);
      }


    } // namespace MatrixFreeFunctions
  }   // namespace internal
} // namespace hyperdeal

#endif
