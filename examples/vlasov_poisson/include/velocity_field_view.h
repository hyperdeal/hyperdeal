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

#ifndef HYPERDEAL_PHASESPACEVELOCITYFIELDVIEW
#define HYPERDEAL_PHASESPACEVELOCITYFIELDVIEW

#include <hyper.deal/matrix_free/id.h>
#include <hyper.deal/operators/advection/velocity_field_view.h>

#include "./derivative_container.h"

namespace hyperdeal
{
  namespace vp
  {
    template <int dim_x,
              int dim_v,
              int degree,
              int n_points,
              typename Number,
              typename VectorizedArrayType>
    class PhaseSpaceVelocityFieldView
      : public hyperdeal::advection::VelocityFieldView<dim_x + dim_v,
                                                       Number,
                                                       TensorID,
                                                       VectorizedArrayType,
                                                       dim_x,
                                                       dim_v>
    {
    public:
      static_assert(dim_x == dim_v,
                    "Vlasov-Poisson only implemented for dim_x == dim_v");

      static const int dim = dim_x + dim_v;

      using MF =
        hyperdeal::MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>;

      using VectorizedArrayTypeX = typename MF::VectorizedArrayTypeX;
      using VectorizedArrayTypeV = typename MF::VectorizedArrayTypeV;

      PhaseSpaceVelocityFieldView(
        const hyperdeal::MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>
          &matrix_free,
        const DerivativeContainer<dim_x,
                                  degree,
                                  n_points,
                                  Number,
                                  VectorizedArrayTypeX>
          &negative_electric_field)
        : evaluator_v(matrix_free.get_matrix_free_v(), 0, /*q-index*/ 0)
        , negative_electric_field(negative_electric_field)
      {}



      void
      reinit(TensorID cell_index) override
      {
        this->index_x   = cell_index.x;
        this->index_v   = cell_index.v / VectorizedArrayTypeV::size();
        this->lane_v    = cell_index.v % VectorizedArrayTypeV::size();
        this->face_type = TensorID::SpaceType::XV;

        evaluator_v.reinit(index_v);
      }



      void
      reinit_face(TensorID face_index) override
      {
        this->index_x   = face_index.x;
        this->index_v   = face_index.v / VectorizedArrayTypeV::size();
        this->lane_v    = face_index.v % VectorizedArrayTypeV::size();
        this->face_type = face_index.type;

        if (face_type == TensorID::SpaceType::X)
          evaluator_v.reinit(index_v);
      }



      void
      reinit_face(TensorID id, unsigned int face) override
      {
        this->reinit(id);

        AssertIndexRange(face, 2 * (dim_x + dim_v));

        if (face < 2 * dim_x)
          this->face_type = TensorID::SpaceType::X;
        else
          this->face_type = TensorID::SpaceType::V;
      }



      inline DEAL_II_ALWAYS_INLINE //
        dealii::Tensor<1, dim_x, VectorizedArrayType>
        evaluate_x(unsigned int /*q*/,
                   unsigned int /*q_x*/,
                   unsigned int q_v) const override
      {
        Assert(face_type == TensorID::SpaceType::XV,
               dealii::StandardExceptions::ExcMessage("No cell given!"));
        AssertIndexRange(q_v, dealii::Utilities::pow(n_points, dim_v));

        return bcast(evaluator_v.quadrature_point(q_v), this->lane_v);
      }



      inline DEAL_II_ALWAYS_INLINE //
        dealii::Tensor<1, dim_v, VectorizedArrayType>
        evaluate_v(unsigned int /*q*/,
                   unsigned int q_x,
                   unsigned int /*q_v*/) const override
      {
        Assert(index_x != dealii::numbers::invalid_unsigned_int,
               dealii::StandardExceptions::ExcMessage(
                 "The function reinit has not been called!"));
        Assert(face_type == TensorID::SpaceType::XV,
               dealii::StandardExceptions::ExcMessage("No cell given!"));
        AssertIndexRange(q_x, dealii::Utilities::pow(n_points, dim_x));

        // TODO: add magnetic field
        return negative_electric_field.get_value_cell(index_x, q_x);
      }



      inline DEAL_II_ALWAYS_INLINE //
        dealii::Tensor<1, dim_x, VectorizedArrayType>
        evaluate_face_x(unsigned int /*q*/,
                        unsigned int /*q_x*/,
                        unsigned int q_v) const override
      {
        Assert(face_type == TensorID::SpaceType::X,
               dealii::StandardExceptions::ExcMessage("No x-face given!"));
        AssertIndexRange(q_v, dealii::Utilities::pow(n_points, dim_v));

        return bcast(evaluator_v.quadrature_point(q_v), this->lane_v);
      }



      inline DEAL_II_ALWAYS_INLINE //
        dealii::Tensor<1, dim_v, VectorizedArrayType>
        evaluate_face_v(unsigned int /*q*/,
                        unsigned int q_x,
                        unsigned int /*q_v*/) const override
      {
        Assert(index_x != dealii::numbers::invalid_unsigned_int,
               dealii::StandardExceptions::ExcMessage(
                 "The function reinit has not been called!"));
        Assert(face_type == TensorID::SpaceType::V,
               dealii::StandardExceptions::ExcMessage("No v-face given!"));
        AssertIndexRange(q_x, dealii::Utilities::pow(n_points, dim_x));

        // TODO: add magnetic field
        return negative_electric_field.get_value_cell(index_x, q_x);
      }



    private:
      static inline DEAL_II_ALWAYS_INLINE //
        dealii::Tensor<1, dim_x, VectorizedArrayTypeX>
        bcast(const dealii::Tensor<1, dim_v, VectorizedArrayTypeV> vector_in,
              const unsigned int                                   lane_v)
      {
        dealii::Tensor<1, dim_x, VectorizedArrayTypeX> vector_out;
        for (unsigned int d = 0; d < dim_x;
             ++d) // if-statment is evaluated at compile time
          vector_out[d] =
            vector_in[d][VectorizedArrayTypeV::size() == 0 ? 0 : lane_v];
        return vector_out;
      }


      // current macro cell/face
      unsigned int index_x = dealii::numbers::invalid_unsigned_int;
      unsigned int index_v = dealii::numbers::invalid_unsigned_int;
      unsigned int lane_v  = dealii::numbers::invalid_unsigned_int;

      typename TensorID::SpaceType face_type = TensorID::SpaceType::XV;

      dealii::
        FEEvaluation<dim_v, degree, n_points, 1, Number, VectorizedArrayTypeV>
                                                       evaluator_v;
      const DerivativeContainer<dim_x,
                                degree,
                                n_points,
                                Number,
                                VectorizedArrayTypeX> &negative_electric_field;
    };
  } // namespace vp
} // namespace hyperdeal

#endif
