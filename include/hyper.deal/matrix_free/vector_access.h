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

#ifndef DEALII_LINEARALGEBRA_SHAREDMPI_EVALUATION
#define DEALII_LINEARALGEBRA_SHAREDMPI_EVALUATION

#include <hyper.deal/base/config.h>

#include <hyper.deal/matrix_free/dof_info.h>
#include <hyper.deal/matrix_free/face_info.h>
#include <hyper.deal/matrix_free/vector_access_kernels.h>

namespace hyperdeal
{
  namespace internal
  {
    namespace MatrixFreeFunctions
    {
      /**
       * Read/write functions translating macro IDs to addresses.
       *
       * TODO: move to deal.II or make it similar to VectorReader/VectorWriter.
       */
      template <typename Number>
      class VectorReaderWriter
      {
      public:
        VectorReaderWriter(
          const hyperdeal::internal::MatrixFreeFunctions::DoFInfo & dof_info,
          const hyperdeal::internal::MatrixFreeFunctions::FaceInfo &face_info);

        template <int dim, int degree, typename VectorizedArrayType>
        void
        read_dof_values_cell_batched(
          const std::vector<double *> &data_others,
          Number *                     dst,
          const unsigned int           cell_batch_number) const;


        template <int dim, int degree, typename VectorizedArrayType>
        void
        read_dof_values_cell_batched(const std::vector<double *> &data_others,
                                     Number *                     dst,
                                     const unsigned int face_batch_number,
                                     const unsigned int face_no,
                                     const unsigned int side) const;


        template <int dim, int degree, typename VectorizedArrayType>
        void
        distribute_local_to_global_cell_batched(
          std::vector<double *> &data_others,
          const Number *         src,
          const unsigned int     cell_batch_number) const;


        template <int dim, int degree, typename VectorizedArrayType>
        void
        distribute_local_to_global_cell_batched(
          std::vector<double *> &data_others,
          const Number *         src,
          const unsigned int     face_batch_number,
          const unsigned int     face_no,
          const unsigned int     side) const;


        template <int dim, int degree, typename VectorizedArrayType>
        void
        set_dof_values_cell_batched(std::vector<double *> &data_others,
                                    const Number *         src,
                                    const unsigned int cell_batch_number) const;


        template <int dim, int degree, typename VectorizedArrayType>
        void
        read_dof_values_face_batched(const std::vector<double *> &data_others,
                                     Number *                     dst,
                                     const unsigned int face_batch_number,
                                     const unsigned int face_no,
                                     const unsigned int side) const;


        template <int dim, int degree, typename VectorizedArrayType>
        void
        distribute_local_to_global_face_batched(
          std::vector<double *> &data_others,
          const Number *         src,
          const unsigned int     face_batch_number,
          const unsigned int     face_no,
          const unsigned int     side) const;


        const std::array<std::vector<unsigned char>, 4>
          &n_vectorization_lanes_filled;
        const std::array<std::vector<std::pair<unsigned int, unsigned int>>, 4>
          &                                     dof_indices_contiguous_ptr;
        const std::array<std::vector<bool>, 4> &face_type;
        const std::array<std::vector<bool>, 4> &face_all;
      };



      template <typename Number>
      VectorReaderWriter<Number>::VectorReaderWriter(
        const hyperdeal::internal::MatrixFreeFunctions::DoFInfo & dof_info,
        const hyperdeal::internal::MatrixFreeFunctions::FaceInfo &face_info)
        : n_vectorization_lanes_filled(dof_info.n_vectorization_lanes_filled)
        , dof_indices_contiguous_ptr(dof_info.dof_indices_contiguous_ptr)
        , face_type(face_info.face_type)
        , face_all(face_info.face_all)
      {}



      template <typename Number>
      template <int dim, int degree, typename VectorizedArrayType>
      void
      VectorReaderWriter<Number>::read_dof_values_cell_batched(
        const std::vector<double *> &data_others,
        Number *                     dst,
        const unsigned int           cell_batch_number) const
      {
        using DA =
          VectorReaderWriterKernels<dim, degree, Number, VectorizedArrayType>;
        static const int v_len = VectorizedArrayType::size();

        if (n_vectorization_lanes_filled[2][cell_batch_number] == v_len)
          {
            std::array<Number *, v_len> srcs;
            for (unsigned int v = 0; v < v_len; v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[2][v_len * cell_batch_number + v];
                srcs[v] = data_others[i.first] + i.second;
              }
            DA::gatherv(srcs, dst);
          }
        else
          {
            for (unsigned int v = 0;
                 v < n_vectorization_lanes_filled[2][cell_batch_number] &&
                 v < v_len;
                 v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[2][v_len * cell_batch_number + v];
                DA::template gather<v_len>(data_others[i.first] + i.second,
                                           dst + v);
              }
          }
      }

      template <typename Number>
      template <int dim, int degree, typename VectorizedArrayType>
      void
      VectorReaderWriter<Number>::read_dof_values_cell_batched(
        const std::vector<double *> &data_others,
        Number *                     dst,
        const unsigned int           face_batch_number,
        const unsigned int /*face_no*/,
        const unsigned int side) const
      {
        using DA =
          VectorReaderWriterKernels<dim, degree, Number, VectorizedArrayType>;
        static const int v_len = VectorizedArrayType::size();

        if (n_vectorization_lanes_filled[side][face_batch_number] == v_len &&
            face_all[side][face_batch_number])
          {
            std::array<Number *, v_len> srcs;
            for (unsigned int v = 0; v < v_len; v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[side]
                                            [v_len * face_batch_number + v];
                srcs[v] = data_others[i.first] + i.second;
              }
            DA::gatherv(srcs, dst);
          }
        else
          {
            for (unsigned int v = 0;
                 v < n_vectorization_lanes_filled[side][face_batch_number] &&
                 v < v_len;
                 v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[side]
                                            [v_len * face_batch_number + v];
                DA::template gather<v_len>(data_others[i.first] + i.second,
                                           dst + v);
              }
          }
      }

      template <typename Number>
      template <int dim, int degree, typename VectorizedArrayType>
      void
      VectorReaderWriter<Number>::read_dof_values_face_batched(
        const std::vector<double *> &data_others,
        Number *                     dst,
        const unsigned int           face_batch_number,
        const unsigned int           face_no,
        const unsigned int           side) const
      {
        using DA =
          VectorReaderWriterKernels<dim, degree, Number, VectorizedArrayType>;
        static const int v_len = VectorizedArrayType::size();

        Assert(side < 4,
               dealii::ExcMessage(
                 "Size of n_vectorization_lanes_filled does not match (1)."));
        Assert(face_batch_number < n_vectorization_lanes_filled[side].size(),
               dealii::ExcMessage(
                 "Size of n_vectorization_lanes_filled does not match (2)."));
        Assert(face_batch_number < face_all[side].size(),
               dealii::ExcMessage(
                 "Size of n_vectorization_lanes_filled does not match (2)."));

        if (n_vectorization_lanes_filled[side][face_batch_number] == v_len &&
            face_all[side][face_batch_number])
          {
            std::array<Number *, v_len> srcs;
            for (unsigned int v = 0; v < v_len; v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[side]
                                            [v_len * face_batch_number + v];
                srcs[v] = data_others[i.first] + i.second;
              }
            DA::template gatherv_face<v_len>(
              srcs, face_no, dst, face_type[side][v_len * face_batch_number]);
          }
        else
          {
            for (unsigned int v = 0;
                 v < n_vectorization_lanes_filled[side][face_batch_number] &&
                 v < v_len;
                 v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[side]
                                            [v_len * face_batch_number + v];
                DA::template gather_face<v_len>(
                  data_others[i.first] + i.second,
                  face_no,
                  dst + v,
                  face_type[side][v_len * face_batch_number + v]);
              }
          }
      }

      template <typename Number>
      template <int dim, int degree, typename VectorizedArrayType>
      void
      VectorReaderWriter<Number>::distribute_local_to_global_cell_batched(
        std::vector<double *> &data_others,
        const Number *         src,
        const unsigned int     cell_batch_number) const
      {
        using DA =
          VectorReaderWriterKernels<dim, degree, Number, VectorizedArrayType>;
        static const int v_len = VectorizedArrayType::size();

        if (n_vectorization_lanes_filled[2][cell_batch_number] == v_len)
          {
            std::array<Number *, v_len> dsts;
            for (unsigned int v = 0; v < v_len; v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[2][v_len * cell_batch_number + v];
                dsts[v] = data_others[i.first] + i.second;
              }
            DA::template scatterv<true>(dsts, src);
          }
        else
          {
            for (unsigned int v = 0;
                 v < n_vectorization_lanes_filled[2][cell_batch_number] &&
                 v < v_len;
                 v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[2][v_len * cell_batch_number + v];
                DA::template scatter<v_len, true>(data_others[i.first] +
                                                    i.second,
                                                  src + v);
              }
          }
      }

      template <typename Number>
      template <int dim, int degree, typename VectorizedArrayType>
      void
      VectorReaderWriter<Number>::distribute_local_to_global_cell_batched(
        std::vector<double *> &data_others,
        const Number *         src,
        const unsigned int     face_batch_number,
        const unsigned int /*face_no*/,
        const unsigned int side) const
      {
        using DA =
          VectorReaderWriterKernels<dim, degree, Number, VectorizedArrayType>;
        static const int v_len = VectorizedArrayType::size();

        if (n_vectorization_lanes_filled[side][face_batch_number] == v_len &&
            face_all[side][face_batch_number])
          {
            std::array<Number *, v_len> dsts;
            for (unsigned int v = 0; v < v_len; v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[side]
                                            [v_len * face_batch_number + v];
                dsts[v] = data_others[i.first] + i.second;
              }
            DA::template scatterv<true>(dsts, src);
          }
        else
          {
            for (unsigned int v = 0;
                 v < n_vectorization_lanes_filled[side][face_batch_number] &&
                 v < v_len;
                 v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[side]
                                            [v_len * face_batch_number + v];
                DA::template scatter<v_len, true>(data_others[i.first] +
                                                    i.second,
                                                  src + v);
              }
          }
      }

      template <typename Number>
      template <int dim, int degree, typename VectorizedArrayType>
      void
      VectorReaderWriter<Number>::distribute_local_to_global_face_batched(
        std::vector<double *> &data_others,
        const Number *         src,
        const unsigned int     face_batch_number,
        const unsigned int     face_no,
        const unsigned int     side) const
      {
        using DA =
          VectorReaderWriterKernels<dim, degree, Number, VectorizedArrayType>;
        static const int v_len = VectorizedArrayType::size();

        if (n_vectorization_lanes_filled[side][face_batch_number] == v_len &&
            face_all[side][face_batch_number])
          {
            std::array<Number *, v_len> dsts;
            for (unsigned int v = 0; v < v_len; v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[side]
                                            [v_len * face_batch_number + v];
                dsts[v] = data_others[i.first] + i.second;
              }
            DA::template scatterv_face<v_len, true>(
              dsts, face_no, src, face_type[side][v_len * face_batch_number]);
          }
        else
          {
            for (unsigned int v = 0;
                 v < n_vectorization_lanes_filled[side][face_batch_number] &&
                 v < v_len;
                 v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[side]
                                            [v_len * face_batch_number + v];
                DA::template scatter_face<v_len, true>(
                  data_others[i.first] + i.second,
                  face_no,
                  src + v,
                  face_type[side][v_len * face_batch_number + v]);
              }
          }
      }

      template <typename Number>
      template <int dim, int degree, typename VectorizedArrayType>
      void
      VectorReaderWriter<Number>::set_dof_values_cell_batched(
        std::vector<double *> &data_others,
        const Number *         src,
        const unsigned int     cell_batch_number) const
      {
        using DA =
          VectorReaderWriterKernels<dim, degree, Number, VectorizedArrayType>;
        static const int v_len = VectorizedArrayType::size();

        if (n_vectorization_lanes_filled[2][cell_batch_number] == v_len)
          {
            std::array<Number *, v_len> dsts;
            for (unsigned int v = 0; v < v_len; v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[2][v_len * cell_batch_number + v];
                dsts[v] = data_others[i.first] + i.second;
              }
            DA::template scatterv<false>(dsts, src);
          }
        else
          {
            for (unsigned int v = 0;
                 v < n_vectorization_lanes_filled[2][cell_batch_number] &&
                 v < v_len;
                 v++)
              {
                auto i =
                  dof_indices_contiguous_ptr[2][v_len * cell_batch_number + v];
                DA::template scatter<v_len, false>(data_others[i.first] +
                                                     i.second,
                                                   src + v);
              }
          }
      }

    } // namespace MatrixFreeFunctions
  }   // namespace internal
} // namespace hyperdeal

#endif
