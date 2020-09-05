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

#ifndef HYPERDEAL_MATRIX_FREEE_EVALUATION_KERNELS
#define HYPERDEAL_MATRIX_FREEE_EVALUATION_KERNELS

#include <hyper.deal/base/config.h>

#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>

namespace hyperdeal
{
  namespace internal
  {
    /**
     * Helper class for interpolating values at cell quadrature points to the
     * quadrature points of a given face.
     *
     * TODO: move into deal.II
     */
    template <int dim, int n_rows, typename Number>
    class FEFaceNormalEvaluation
    {
    public:
      /**
       * Constructor.
       */
      FEFaceNormalEvaluation(
        const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number>
          &shape_info)
        : shape_info(shape_info)
      {}

      /**
       * Perform interpolation for face @p face_no.
       */
      template <bool forward, bool add>
      void
      interpolate(Number *           output,
                  const Number *     input,
                  const unsigned int face_no)
      {
        // clang-format off
        if (dim >= 1 && face_no / 2 == 0) interpolate_impl<0, forward, add>(output, input, face_no); else
        if (dim >= 2 && face_no / 2 == 1) interpolate_impl<1, forward, add>(output, input, face_no); else
        if (dim >= 3 && face_no / 2 == 2) interpolate_impl<2, forward, add>(output, input, face_no); else
        if (dim >= 4 && face_no / 2 == 3) interpolate_impl<3, forward, add>(output, input, face_no); else
        if (dim >= 5 && face_no / 2 == 4) interpolate_impl<4, forward, add>(output, input, face_no); else
        if (dim >= 6 && face_no / 2 == 5) interpolate_impl<5, forward, add>(output, input, face_no); else
          {
            Assert(false, dealii::StandardExceptions::ExcNotImplemented());
          }
        // clang-format on
      }

    private:
      /**
       * Do the actural interpolation.
       */
      template <int face_direction, bool contract_onto_face, bool add>
      void
      interpolate_impl(Number *DEAL_II_RESTRICT out,
                       const Number *DEAL_II_RESTRICT in,
                       const unsigned int             face_no) const
      {
        AssertIndexRange(face_direction, dim);

        constexpr auto n_blocks1 =
          dealii::Utilities::pow<unsigned int>(n_rows, face_direction);
        constexpr auto n_blocks2 = dealii::Utilities::pow<unsigned int>(
          n_rows, std::max(dim - face_direction - 1, 0));

        constexpr auto stride =
          dealii::Utilities::pow<unsigned int>(n_rows, face_direction);

        const Number *DEAL_II_RESTRICT shape_values =
          &shape_info.data[0].quadrature_data_on_face[face_no % 2][0];

        for (unsigned int i2 = 0u; i2 < n_blocks2; ++i2)
          {
            for (unsigned int i1 = 0u; i1 < n_blocks1; ++i1)
              {
                if (contract_onto_face)
                  {
                    Number res0 = in[0] * shape_values[0];

                    for (unsigned int ind = 1; ind < n_rows; ++ind)
                      res0 += in[ind * stride] * shape_values[ind];

                    if (add == false)
                      out[0] = res0;
                    else
                      out[0] += res0;
                  }
                else
                  {
                    for (unsigned int col = 0; col < n_rows; ++col)
                      {
                        if (add == false)
                          out[col * stride] = in[0] * shape_values[col];
                        else
                          out[col * stride] += in[0] * shape_values[col];
                      }
                  }

                ++out;
                ++in;
              }

            if (contract_onto_face)
              in += (dealii::Utilities::pow(n_rows, face_direction + 1) -
                     n_blocks1);
            else
              out += (dealii::Utilities::pow(n_rows, face_direction + 1) -
                      n_blocks1);
          }
      }

      // TODO: use the hyper.deal version?
      const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number>
        &shape_info;
    };

  } // namespace internal

} // namespace hyperdeal


#endif
