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

#ifndef HYPERDEAL_FUNCTIONALITIES_VELOCITY_FIELD_VIEW
#define HYPERDEAL_FUNCTIONALITIES_VELOCITY_FIELD_VIEW

#include <hyper.deal/base/config.h>

#include <hyper.deal/matrix_free/id.h>

namespace hyperdeal
{
  namespace advection
  {
    template <int dim,
              typename Number,
              typename ID,
              typename VectorizedArrayType,
              int dim_x = dim,
              int dim_v = dim>
    class VelocityFieldView
    {
    public:
      virtual ~VelocityFieldView() = default;

      virtual void
      reinit(ID id) = 0;

      virtual void
      reinit_face(ID id) = 0;

      virtual void
      reinit_face(ID id, unsigned int face) = 0;

      virtual dealii::Tensor<1, dim_x, VectorizedArrayType>
      evaluate_x(unsigned int q, unsigned int qx, unsigned int qv) const = 0;

      virtual dealii::Tensor<1, dim_v, VectorizedArrayType>
      evaluate_v(unsigned int q, unsigned int qx, unsigned int qv) const = 0;

      virtual dealii::Tensor<1, dim_x, VectorizedArrayType>
      evaluate_face_x(unsigned int q,
                      unsigned int qx,
                      unsigned int qv) const = 0;

      virtual dealii::Tensor<1, dim_v, VectorizedArrayType>
      evaluate_face_v(unsigned int q,
                      unsigned int qx,
                      unsigned int qv) const = 0;
    };

    template <int dim,
              typename Number,
              typename VectorizedArrayType,
              int dim_x,
              int dim_v>
    class ConstantVelocityFieldView
      : public VelocityFieldView<dim,
                                 Number,
                                 TensorID,
                                 VectorizedArrayType,
                                 dim_x,
                                 dim_v>
    {
    public:
      ConstantVelocityFieldView(
        const dealii::Tensor<1, dim, VectorizedArrayType> &transport_direction)
        : transport_direction(transport_direction)
        , transport_direction_x(extract<dim_x>(transport_direction, 0))
        , transport_direction_v(extract<dim_v>(transport_direction, dim_x))
      {}

      void reinit(TensorID /*id*/) override
      {
        // nothing to do
      }

      void reinit_face(TensorID /*id*/) override
      {
        // nothing to do
      }



      void
      reinit_face(TensorID /*id*/, unsigned int /*face*/) override
      {
        // nothing to do
      }



      inline DEAL_II_ALWAYS_INLINE //
        dealii::Tensor<1, dim_x, VectorizedArrayType>
        evaluate_x(unsigned int /*q*/,
                   unsigned int /*qx*/,
                   unsigned int /*qv*/) const override
      {
        return transport_direction_x;
      }



      inline DEAL_II_ALWAYS_INLINE //
        dealii::Tensor<1, dim_v, VectorizedArrayType>
        evaluate_v(unsigned int /*q*/,
                   unsigned int /*qx*/,
                   unsigned int /*qv*/) const override
      {
        return transport_direction_v;
      }



      inline DEAL_II_ALWAYS_INLINE //
        dealii::Tensor<1, dim_x, VectorizedArrayType>
        evaluate_face_x(unsigned int /*q*/,
                        unsigned int /*qx*/,
                        unsigned int /*qv*/) const override
      {
        return transport_direction_x;
      }



      inline DEAL_II_ALWAYS_INLINE //
        dealii::Tensor<1, dim_v, VectorizedArrayType>
        evaluate_face_v(unsigned int /*q*/,
                        unsigned int /*qx*/,
                        unsigned int /*qv*/) const override
      {
        return transport_direction_v;
      }



    private:
      template <int dim_>
      static dealii::Tensor<1, dim_, VectorizedArrayType>
        extract(dealii::Tensor<1, dim, VectorizedArrayType> input,
                unsigned int                                offset)
      {
        dealii::Tensor<1, dim_, VectorizedArrayType> output;

        for (auto i = 0u; i < dim_; i++)
          output[i] = input[offset + i];

        return output;
      }

      const dealii::Tensor<1, dim, VectorizedArrayType>   transport_direction;
      const dealii::Tensor<1, dim_x, VectorizedArrayType> transport_direction_x;
      const dealii::Tensor<1, dim_v, VectorizedArrayType> transport_direction_v;
    };

  } // namespace advection
} // namespace hyperdeal

#endif
