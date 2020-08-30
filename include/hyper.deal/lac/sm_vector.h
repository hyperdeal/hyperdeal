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

#ifndef DEALII_LINEARALGEBRA_SHAREDMPI_VECTOR
#define DEALII_LINEARALGEBRA_SHAREDMPI_VECTOR

#include <hyper.deal/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/partitioner.h>

#include <deal.II/lac/vector_operation.h>

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace SharedMPI
  {
    /**
     * Vector class built around MPI-3.0 shared-memory features.
     *
     * @note Consider this class only as a container of values (locally owned
     *   and ghosted degrees of freedom). It only has very basic functionality
     *   on top of them. The functionality of updating ghost values and
     *   the interpretation of the degrees of freedom as values of cells or
     *   faces are done by specialized classed externally.
     */
    template <typename Number>
    class Vector
    {
    public:
      using This       = Vector<Number>;
      using value_type = Number;
      using size_type  = std::size_t; // 64-bit -> 1.8e19 :)

      /**
       * Destructor.
       */
      ~Vector();

      /**
       * Initialize vector.
       *
       * @note Not implemented. Use the other one.
       */
      void
      reinit(const This &other, const bool omit_zeroing_entries = false);

      /**
       * Initialize vector with a given partitioner and vector reader.
       */
      void
      reinit(const MPI_Comm comm_all,
             const MPI_Comm comm_shared,
             const int      _local_size,
             const int      _ghost_size);

      /**
       * Get read-write pointer to the beginning of the local data.
       */
      Number *
      begin();

      /**
       * Get read-only pointer to the beginning of the local data.
       */
      const Number *
      begin() const;

      /**
       * Zero out ghost values (same as below).
       */
      void
      reset_ghost_values() const;

      /**
       * Zero out vector.
       */
      void
      zero_out(const bool clear_ghosts = false);

      /**
       * Zero out ghost values.
       */
      void
      zero_out_ghosts() const;

      /**
       * Get const pointers to the beginning of the values of the other
       * processes of the same shared-memory domain.
       *
       * TODO: name of the function?
       */
      std::vector<Number *> &
      other_values();

      /**
       * Get pointers to the beginning of the values of the other
       * processes of the same shared-memory domain.
       *
       * TODO: name of the function?
       */
      const std::vector<Number *> &
      other_values() const;

      /**
       * Return read-write access to local value.
       *
       * @note In contrast to the deal.II implementation it takes the
       *   local index and not the global index [TODO].
       */
      Number &operator[](const size_type local_index);

      /**
       * Return read access to local value.
       *
       * @note In contrast to the deal.II implementation it takes the
       *   local index and not the global index [TODO].
       */
      Number operator[](const size_type local_index) const;

      /**
       * Return read-write access to local value.
       */
      inline DEAL_II_ALWAYS_INLINE //
        Number &
        local_element(const size_type local_index);

      /**
       * Return read access to local value.
       */
      inline DEAL_II_ALWAYS_INLINE //
        Number
        local_element(const size_type local_index) const;

      /**
       * Global size of vector. Sum of local_size() over all processes.
       */
      dealii::types::global_dof_index
      size() const;

      /**
       * Return number of local entries.
       */
      size_type
      local_size() const;

      /**
       * Return number of ghost entries.
       */
      size_type
      n_ghost_entries() const;

      /**
       * Swap underlying vectors for local values and ghost values.
       *
       * @note The partitioner does not change.
       */
      void
      swap(This &other);

      /**
       * Set all local values to s: u[i] = s
       */
      This &
      operator=(const Number s);

      /**
       * Copy local values from other vector: u[i] = v[i]
       */
      This &
      operator=(const This &other);

      /**
       * Add scalar: u[i] += V
       */
      void
      add(const Number &V);

      /**
       * Add scaled other vector element-wise: u[i] += V * v[i]
       */
      void
      add(const Number &V, const This &other);

      /**
       * Element-wise scaled-add: u[i] = s * u[i] a * v[i]
       */
      void
      sadd(const Number s, const Number a, const This &other);

      /**
       * Compute l2-norm.
       */
      Number
      l2_norm() const;

      /**
       * Copy local values from other vector (guarded by two MPI_Barriers in
       * shared-memory domain).
       */
      template <typename T>
      void
      copy_from(T &other);

      /**
       * Copy local values to other vector (guarded by two MPI_Barriers in
       * shared-memory domain).
       */
      template <typename T>
      void
      copy_to(T &other) const;

      /**
       * Print values and ghost values of vector.
       */
      void
      print(std::ostream &     out,
            const unsigned int precision  = 3,
            const bool         scientific = true,
            const bool         across     = true) const;

      /**
       * Start and finish ghost value update.
       *
       * @note Not implemented! hyperdeal::MatrixFree will do the work!
       */
      void
      update_ghost_values() const;

      /**
       * Start ghost value update: delegates work to the partitioner.
       *
       * @note Not implemented! hyperdeal::MatrixFree will do the work!
       */
      void
      update_ghost_values_start(
        const unsigned int communication_channel = 0) const;

      /**
       * Finish ghost value update: delegates work to the partitioner.
       *
       * @note Not implemented! hyperdeal::MatrixFree will do the work!
       */
      void
      update_ghost_values_finish() const;

      /**
       * Start and finish compress.
       *
       * @note Not implemented! hyperdeal::MatrixFree will do the work!
       */
      void
      compress(VectorOperation::values operation = VectorOperation::add);

      /**
       * Start compress: delegates work to the partitioner.
       *
       * @note Not implemented! hyperdeal::MatrixFree will do the work!
       */
      void
      compress_start(const unsigned int      communication_channel = 0,
                     VectorOperation::values operation = VectorOperation::add);

      /**
       * Finish compress: delegates work to the partitioner.
       *
       * @note Not implemented! hyperdeal::MatrixFree will do the work!
       */
      void
      compress_finish(VectorOperation::values operation = VectorOperation::add);

      /**
       * Return memory consumption (partitioner not counted).
       *
       * @note Not implemented! hyperdeal::MatrixFree will do the work!
       */
      virtual std::size_t
      memory_consumption() const;

    private:
      /**
       * Actual local data.
       */
      mutable Number *data_this;

      /**
       * Pointers to the local data of the other processes of the same
       * shared-memory domain.
       */
      mutable std::vector<Number *> data_others;

      /**
       * Global communicator.
       */
      MPI_Comm comm_all;

      /**
       * Shared-memory communicator.
       */
      MPI_Comm comm_shared;

      /**
       * Number of locally-owned dofs.
       */
      unsigned int _local_size = 0;

      /**
       * Number of ghost dofs.
       */
      unsigned int _ghost_size = 0;

      /**
       * Configured with buffers within the shared memory?
       *
       * TODO: needed here?
       */
      bool do_buffering;

      /**
       * MPI-window for managing of the shared memory. Only used for
       * allocation and deallocation.
       */
      MPI_Win *wins = nullptr;
    };


    template <typename Number>
    Number
    Vector<Number>::local_element(const size_type local_index) const
    {
      return data_this[local_index];
    }



    template <typename Number>
    Number &
    Vector<Number>::local_element(const size_type local_index)
    {
      return data_this[local_index];
    }



    template <typename Number>
    std::vector<Number *> &
    Vector<Number>::other_values()
    {
      return data_others;
    }



    template <typename Number>
    const std::vector<Number *> &
    Vector<Number>::other_values() const
    {
      return data_others;
    }

  } // namespace SharedMPI
} // namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE

#endif
