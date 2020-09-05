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
    template <unsigned int dim, unsigned int n_points, typename Number>
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
      template <bool forward>
      void
      interpolate(Number *           output,
                  const Number *     input,
                  const unsigned int face_no)
      {
        // clang-format off
        if (dim >= 1 && face_no / 2 == 0) interpolate_impl<0, forward>(output, input, face_no); else
        if (dim >= 2 && face_no / 2 == 1) interpolate_impl<1, forward>(output, input, face_no); else
        if (dim >= 3 && face_no / 2 == 2) interpolate_impl<2, forward>(output, input, face_no); else
        if (dim >= 4 && face_no / 2 == 3) interpolate_impl<3, forward>(output, input, face_no); else
        if (dim >= 5 && face_no / 2 == 4) interpolate_impl<4, forward>(output, input, face_no); else
        if (dim >= 6 && face_no / 2 == 5) interpolate_impl<5, forward>(output, input, face_no); else
          {
            Assert(false, dealii::StandardExceptions::ExcNotImplemented());
          }
        // clang-format on
      }

    private:
      /**
       * Do the actural interpolation.
       */
      template <unsigned int d, bool forward>
      void
      interpolate_impl(Number *           output,
                       const Number *     input,
                       const unsigned int face_no) const
      {
        const auto weights =
          &shape_info.data[0].quadrature_data_on_face[face_no % 2][0];

        for (auto i = 0u, e = 0u;
             i < dealii::Utilities::pow(n_points, dim - d - 1);
             i++)
          for (auto j = 0u; j < dealii::Utilities::pow(n_points, d); j++, e++)
            {
              if (forward)
                output[e] = 0;

              for (auto k = 0u; k < n_points; k++)
                if (forward)
                  output[e] +=
                    input[i * dealii::Utilities::pow(n_points, d + 1) +
                          k * dealii::Utilities::pow(n_points, d) + j] *
                    weights[k];
                else
                  output[i * dealii::Utilities::pow(n_points, d + 1) +
                         k * dealii::Utilities::pow(n_points, d) + j] +=
                    input[e] * weights[k];
            }
      }

      // TODO: use the hyper.deal version?
      const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number>
        &shape_info;
    };

  } // namespace internal

} // namespace hyperdeal


#endif
