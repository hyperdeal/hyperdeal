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

#include <hyper.deal/lac/sm_vector.h>

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace SharedMPI
  {
    template <typename Number>
    Vector<Number>::~Vector()
    {
      if (wins != nullptr)
        MPI_Win_free(wins);
    }

    template <typename Number>
    Number *
    Vector<Number>::begin()
    {
      return data_this;
    }



    template <typename Number>
    const Number *
    Vector<Number>::begin() const
    {
      return data_this;
    }



    template <typename Number>
    Vector<Number> &
    Vector<Number>::operator=(const Number s)
    {
      for (unsigned int i = 0; i < _local_size; i++)
        data_this[i] = s;
      return *this;
    }



    template <typename Number>
    Vector<Number> &
    Vector<Number>::operator=(const Vector<Number> &other)
    {
      for (unsigned int i = 0; i < _local_size; i++)
        data_this[i] = other.data_this[i];
      return *this;
    }



    template <typename Number>
    Number Vector<Number>::operator[](const size_type local_index) const
    {
      return data_this[local_index];
    }



    template <typename Number>
    Number &Vector<Number>::operator[](const size_type local_index)
    {
      return data_this[local_index];
    }



    template <typename Number>
    dealii::types::global_dof_index
    Vector<Number>::size() const
    {
      return dealii::Utilities::MPI::sum(
        (dealii::types::global_dof_index)_local_size, comm_all);
    }



    template <typename Number>
    std::size_t
    Vector<Number>::local_size() const
    {
      return _local_size;
    }



    template <typename Number>
    std::size_t
    Vector<Number>::n_ghost_entries() const
    {
      return _ghost_size;
    }



    template <typename Number>
    void
    Vector<Number>::swap(Vector<Number> &other)
    {
      MPI_Barrier(comm_shared);
      std::swap(this->data_this, other.data_this);
      std::swap(this->data_others, other.data_others);
      MPI_Barrier(comm_shared);
    }



    template <typename Number>
    void
    Vector<Number>::add(const Number &V)
    {
      for (unsigned int i = 0; i < _local_size; i++)
        data_this[i] += V;
    }



    template <typename Number>
    void
    Vector<Number>::add(const Number &V, const Vector<Number> &other)
    {
      for (unsigned int i = 0; i < _local_size; i++)
        data_this[i] += other.data_this[i] * V;
    }



    template <typename Number>
    void
    Vector<Number>::sadd(const Number          s,
                         const Number          a,
                         const Vector<Number> &other)
    {
      for (unsigned int i = 0; i < _local_size; i++)
        data_this[i] = data_this[i] * s + other.data_this[i] * a;
    }



    template <typename Number>
    void
    Vector<Number>::reinit(const Vector<Number> &other,
                           const bool            omit_zeroing_entries)
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented yet."));
      (void)other;
      (void)omit_zeroing_entries;
    }



    template <typename Number>
    void
    Vector<Number>::print(std::ostream &     out,
                          const unsigned int precision,
                          const bool         scientific,
                          const bool         across) const
    {
      // TODO: Currently implemented as in deal.II. Replace this by a version
      // in which only the root process outputs.

      std::ios::fmtflags old_flags     = out.flags();
      unsigned int       old_precision = out.precision(precision);

      out.precision(precision);
      if (scientific)
        out.setf(std::ios::scientific, std::ios::floatfield);
      else
        out.setf(std::ios::fixed, std::ios::floatfield);

      // to make the vector write out all the information in order, use as
      // many barriers as there are processors and start writing when it's our
      // turn
      if (Utilities::MPI::n_mpi_processes(this->comm_all) > 1)
        for (unsigned int i = 0;
             i < Utilities::MPI::this_mpi_process(this->comm_all);
             i++)
          {
            const int ierr = MPI_Barrier(comm_all);
            AssertThrowMPI(ierr);
          }

      out << "Local:" << std::endl;
      if (across)
        for (unsigned int i = 0; i < _local_size; ++i)
          out << data_this[i] << ' ';
      else
        for (unsigned int i = 0; i < _local_size; ++i)
          out << data_this[i] << std::endl;
      out << std::endl;

      out << "Ghost:" << std::endl;
      if (across)
        for (unsigned int i = _local_size; i < _local_size + _ghost_size; ++i)
          out << data_this[i] << ' ';
      else
        for (unsigned int i = _ghost_size; i < _local_size + _ghost_size; ++i)
          out << data_this[i] << std::endl;
      out << std::endl << std::endl;

      out << std::flush;

      if (Utilities::MPI::n_mpi_processes(this->comm_all) > 1)
        {
          int ierr = MPI_Barrier(comm_all);
          AssertThrowMPI(ierr);

          for (unsigned int i =
                 Utilities::MPI::this_mpi_process(this->comm_all) + 1;
               i < Utilities::MPI::n_mpi_processes(this->comm_all);
               i++)
            {
              ierr = MPI_Barrier(comm_all);
              AssertThrowMPI(ierr);
            }
        }

      AssertThrow(out, dealii::ExcIO());
      // reset output format
      out.flags(old_flags);
      out.precision(old_precision);
    }



    template <typename Number>
    void
    Vector<Number>::update_ghost_values() const
    {
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
    }



    template <typename Number>
    void
    Vector<Number>::update_ghost_values_start(
      const unsigned int communication_channel) const
    {
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());

      (void)communication_channel;
    }



    template <typename Number>
    void
    Vector<Number>::update_ghost_values_finish() const
    {
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());
    }



    template <typename Number>
    void
    Vector<Number>::compress(VectorOperation::values operation)
    {
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());

      (void)operation;
    }



    template <typename Number>
    void
    Vector<Number>::compress_start(const unsigned int communication_channel,
                                   VectorOperation::values operation)
    {
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());

      (void)communication_channel;
      (void)operation;
    }



    template <typename Number>
    void
    Vector<Number>::compress_finish(VectorOperation::values operation)
    {
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());

      (void)operation;
    }



    template <typename Number>
    void
    Vector<Number>::reset_ghost_values() const
    {
      this->zero_out_ghosts();
    }



    template <typename Number>
    void
    Vector<Number>::reinit(const MPI_Comm comm_all,
                           const MPI_Comm comm_shared,
                           const int      _local_size,
                           const int      _ghost_size)
    {
      AssertThrow(wins == nullptr,
                  ExcMessage("Vector has been already initialized!"));

      // cache frequently used variables
      this->comm_all    = comm_all;
      this->comm_shared = comm_shared;
      this->_local_size = _local_size;
      this->_ghost_size = _ghost_size;

      const unsigned int size_shared =
        Utilities::MPI::n_mpi_processes(this->comm_shared);
      const unsigned int rank_shared =
        Utilities::MPI::this_mpi_process(this->comm_shared);


      data_this = (Number *)malloc(0);
      data_others.resize(size_shared);

      wins = new MPI_Win;

      MPI_Info info;
      MPI_Info_create(&info);
      MPI_Info_set(info, "alloc_shared_noncontig", "true");

      MPI_Win_allocate_shared((_local_size + _ghost_size) * sizeof(Number),
                              sizeof(Number),
                              info,
                              comm_shared,
                              data_this,
                              wins);

      for (unsigned int i = 0; i < size_shared; i++)
        {
          int      disp_unit;
          MPI_Aint ssize;
          MPI_Win_shared_query(*wins, i, &ssize, &disp_unit, &data_others[i]);
        }

      data_this = data_others[rank_shared];
    }



    template <typename Number>
    Number
    Vector<Number>::l2_norm() const
    {
      Number temp_local  = 0.0;
      Number temp_global = 0.0;

      for (unsigned int i = 0; i < _local_size; i++)
        temp_local += data_this[i] * data_this[i];

      MPI_Allreduce(
        &temp_local, &temp_global, 1, MPI_DOUBLE, MPI_SUM, comm_all);

      return std::sqrt(temp_global);
    }



    template <typename Number>
    void
    Vector<Number>::zero_out_ghosts() const
    {
      for (unsigned int i = _local_size; i < _local_size + _ghost_size; i++)
        data_this[i] = 0.0;
    }



    template <typename Number>
    template <typename T>
    void
    Vector<Number>::copy_to(T &other) const
    {
      MPI_Barrier(comm_shared);

      // TOOD: assert that partitioners are compatible

      for (int i = 0; i < _local_size; i++)
        other.local_element(i) = data_this[i];

      MPI_Barrier(comm_shared);
    }



    template <typename Number>
    template <typename T>
    void
    Vector<Number>::copy_from(T &other)
    {
      MPI_Barrier(comm_shared);

      // TOOD: assert that partitioners are compatible

      for (int i = 0; i < _local_size; i++)
        data_this[i] = other.local_element(i);

      MPI_Barrier(comm_shared);
    }



    template <typename Number>
    std::size_t
    Vector<Number>::memory_consumption() const
    {
      return (_local_size + _ghost_size) * sizeof(Number);
    }



    template <typename Number>
    void
    Vector<Number>::zero_out(const bool clear_ghosts)
    {
      for (unsigned int i = 0;
           i < _local_size + (clear_ghosts ? _ghost_size : 0);
           i++)
        data_this[i] = 0.0;
    }



    template class Vector<double>;

  } // namespace SharedMPI
} // namespace LinearAlgebra

DEAL_II_NAMESPACE_CLOSE