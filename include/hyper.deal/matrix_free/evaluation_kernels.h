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
          &shape_info.data[0].shape_data_on_face[face_no % 2][0];

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

    /***
     * This class is a copy from deal.II. It has been amended with the function
     * do_backward_hessians().
     *
     * TODO: move into deal.II
     */
    template <dealii::internal::EvaluatorVariant variant,
              int                                dim,
              int                                basis_size_1,
              int                                basis_size_2,
              int                                n_components,
              typename Number,
              typename Number2>
    struct FEEvaluationImplBasisChange
    {
      static_assert(basis_size_1 == 0 || basis_size_1 <= basis_size_2,
                    "The second dimension must not be smaller than the first");

#ifndef DEBUG
      DEAL_II_ALWAYS_INLINE
#endif
      static void
      do_forward(const dealii::AlignedVector<Number2> &transformation_matrix,
                 const Number *                        values_in,
                 Number *                              values_out,
                 const unsigned int                    basis_size_1_variable =
                   dealii::numbers::invalid_unsigned_int,
                 const unsigned int basis_size_2_variable =
                   dealii::numbers::invalid_unsigned_int)
      {
        Assert(basis_size_1 != 0 ||
                 basis_size_1_variable <= basis_size_2_variable,
               dealii::ExcMessage(
                 "The second dimension must not be smaller than the first"));

        // we do recursion until dim==1 or dim==2 and we have
        // basis_size_1==basis_size_2. The latter optimization increases
        // optimization possibilities for the compiler but does only work for
        // aliased pointers if the sizes are equal.
        constexpr int next_dim =
          (dim > 2 ||
           ((basis_size_1 == 0 || basis_size_2 > basis_size_1) && dim > 1)) ?
            dim - 1 :
            dim;

        dealii::internal::EvaluatorTensorProduct<
          variant,
          dim,
          basis_size_1,
          (basis_size_1 == 0 ? 0 : basis_size_2),
          Number,
          Number2>
                           eval_val(transformation_matrix,
                   dealii::AlignedVector<Number2>(),
                   dealii::AlignedVector<Number2>(),
                   basis_size_1_variable,
                   basis_size_2_variable);
        const unsigned int np_1 =
          basis_size_1 > 0 ? basis_size_1 : basis_size_1_variable;
        const unsigned int np_2 =
          basis_size_1 > 0 ? basis_size_2 : basis_size_2_variable;
        Assert(np_1 > 0 && np_1 != dealii::numbers::invalid_unsigned_int,
               dealii::ExcMessage("Cannot transform with 0-point basis"));
        Assert(np_2 > 0 && np_2 != dealii::numbers::invalid_unsigned_int,
               dealii::ExcMessage("Cannot transform with 0-point basis"));

        // run loop backwards to ensure correctness if values_in aliases with
        // values_out in case with basis_size_1 < basis_size_2
        values_in =
          values_in + n_components * dealii::Utilities::fixed_power<dim>(np_1);
        values_out =
          values_out + n_components * dealii::Utilities::fixed_power<dim>(np_2);
        for (unsigned int c = n_components; c != 0; --c)
          {
            values_in -= dealii::Utilities::fixed_power<dim>(np_1);
            values_out -= dealii::Utilities::fixed_power<dim>(np_2);
            if (next_dim < dim)
              for (unsigned int q = np_1; q != 0; --q)
                FEEvaluationImplBasisChange<variant,
                                            next_dim,
                                            basis_size_1,
                                            basis_size_2,
                                            1,
                                            Number,
                                            Number2>::
                  do_forward(transformation_matrix,
                             values_in +
                               (q - 1) *
                                 dealii::Utilities::fixed_power<next_dim>(np_1),
                             values_out +
                               (q - 1) *
                                 dealii::Utilities::fixed_power<next_dim>(np_2),
                             basis_size_1_variable,
                             basis_size_2_variable);

            // the recursion stops if dim==1 or if dim==2 and
            // basis_size_1==basis_size_2 (the latter is used because the
            // compiler generates nicer code)
            if (basis_size_1 > 0 && basis_size_2 == basis_size_1 && dim == 2)
              {
                eval_val.template values<0, true, false>(values_in, values_out);
                eval_val.template values<1, true, false>(values_out,
                                                         values_out);
              }
            else if (dim == 1)
              eval_val.template values<dim - 1, true, false>(values_in,
                                                             values_out);
            else
              eval_val.template values<dim - 1, true, false>(values_out,
                                                             values_out);
          }
      }

#ifndef DEBUG
      DEAL_II_ALWAYS_INLINE
#endif
      static void
      do_backward(const dealii::AlignedVector<Number2> &transformation_matrix,
                  const bool                            add_into_result,
                  Number *                              values_in,
                  Number *                              values_out,
                  const unsigned int                    basis_size_1_variable =
                    dealii::numbers::invalid_unsigned_int,
                  const unsigned int basis_size_2_variable =
                    dealii::numbers::invalid_unsigned_int)
      {
        Assert(basis_size_1 != 0 ||
                 basis_size_1_variable <= basis_size_2_variable,
               dealii::ExcMessage(
                 "The second dimension must not be smaller than the first"));
        Assert(add_into_result == false || values_in != values_out,
               dealii::ExcMessage(
                 "Input and output cannot alias with each other when "
                 "adding the result of the basis change to existing data"));

        constexpr int next_dim =
          (dim > 2 ||
           ((basis_size_1 == 0 || basis_size_2 > basis_size_1) && dim > 1)) ?
            dim - 1 :
            dim;
        dealii::internal::EvaluatorTensorProduct<
          variant,
          dim,
          basis_size_1,
          (basis_size_1 == 0 ? 0 : basis_size_2),
          Number,
          Number2>
                           eval_val(transformation_matrix,
                   dealii::AlignedVector<Number2>(),
                   dealii::AlignedVector<Number2>(),
                   basis_size_1_variable,
                   basis_size_2_variable);
        const unsigned int np_1 =
          basis_size_1 > 0 ? basis_size_1 : basis_size_1_variable;
        const unsigned int np_2 =
          basis_size_1 > 0 ? basis_size_2 : basis_size_2_variable;
        Assert(np_1 > 0 && np_1 != dealii::numbers::invalid_unsigned_int,
               dealii::ExcMessage("Cannot transform with 0-point basis"));
        Assert(np_2 > 0 && np_2 != dealii::numbers::invalid_unsigned_int,
               dealii::ExcMessage("Cannot transform with 0-point basis"));

        for (unsigned int c = 0; c < n_components; ++c)
          {
            if (basis_size_1 > 0 && basis_size_2 == basis_size_1 && dim == 2)
              {
                eval_val.template values<1, false, false>(values_in, values_in);
                if (add_into_result)
                  eval_val.template values<0, false, true>(values_in,
                                                           values_out);
                else
                  eval_val.template values<0, false, false>(values_in,
                                                            values_out);
              }
            else
              {
                if (dim == 1 && add_into_result)
                  eval_val.template values<0, false, true>(values_in,
                                                           values_out);
                else if (dim == 1)
                  eval_val.template values<0, false, false>(values_in,
                                                            values_out);
                else
                  eval_val.template values<dim - 1, false, false>(values_in,
                                                                  values_in);
              }
            if (next_dim < dim)
              for (unsigned int q = 0; q < np_1; ++q)
                FEEvaluationImplBasisChange<variant,
                                            next_dim,
                                            basis_size_1,
                                            basis_size_2,
                                            1,
                                            Number,
                                            Number2>::
                  do_backward(
                    transformation_matrix,
                    add_into_result,
                    values_in +
                      q * dealii::Utilities::fixed_power<next_dim>(np_2),
                    values_out +
                      q * dealii::Utilities::fixed_power<next_dim>(np_1),
                    basis_size_1_variable,
                    basis_size_2_variable);

            values_in += dealii::Utilities::fixed_power<dim>(np_2);
            values_out += dealii::Utilities::fixed_power<dim>(np_1);
          }
      }

#ifndef DEBUG
      DEAL_II_ALWAYS_INLINE
#endif
      static void
      do_backward_hessians(
        const dealii::AlignedVector<Number2> &transformation_matrix,
        const bool                            add_into_result,
        Number *                              values_in,
        Number *                              values_out,
        const unsigned int                    basis_size_1_variable =
          dealii::numbers::invalid_unsigned_int,
        const unsigned int basis_size_2_variable =
          dealii::numbers::invalid_unsigned_int)
      {
        Assert(basis_size_1 != 0 ||
                 basis_size_1_variable <= basis_size_2_variable,
               dealii::ExcMessage(
                 "The second dimension must not be smaller than the first"));
        Assert(add_into_result == false || values_in != values_out,
               dealii::ExcMessage(
                 "Input and output cannot alias with each other when "
                 "adding the result of the basis change to existing data"));

        constexpr int next_dim =
          (dim > 2 ||
           ((basis_size_1 == 0 || basis_size_2 > basis_size_1) && dim > 1)) ?
            dim - 1 :
            dim;
        dealii::internal::EvaluatorTensorProduct<
          variant,
          dim,
          basis_size_1,
          (basis_size_1 == 0 ? 0 : basis_size_2),
          Number,
          Number2>
                           eval_val(dealii::AlignedVector<Number2>(),
                   dealii::AlignedVector<Number2>(),
                   transformation_matrix,
                   basis_size_1_variable,
                   basis_size_2_variable);
        const unsigned int np_1 =
          basis_size_1 > 0 ? basis_size_1 : basis_size_1_variable;
        const unsigned int np_2 =
          basis_size_1 > 0 ? basis_size_2 : basis_size_2_variable;
        Assert(np_1 > 0 && np_1 != dealii::numbers::invalid_unsigned_int,
               dealii::ExcMessage("Cannot transform with 0-point basis"));
        Assert(np_2 > 0 && np_2 != dealii::numbers::invalid_unsigned_int,
               dealii::ExcMessage("Cannot transform with 0-point basis"));

        for (unsigned int c = 0; c < n_components; ++c)
          {
            if (basis_size_1 > 0 && basis_size_2 == basis_size_1 && dim == 2)
              {
                eval_val.template hessians<1, false, false>(values_in,
                                                            values_in);
                if (add_into_result)
                  eval_val.template hessians<0, false, true>(values_in,
                                                             values_out);
                else
                  eval_val.template hessians<0, false, false>(values_in,
                                                              values_out);
              }
            else
              {
                if (dim == 1 && add_into_result)
                  eval_val.template hessians<0, false, true>(values_in,
                                                             values_out);
                else if (dim == 1)
                  eval_val.template hessians<0, false, false>(values_in,
                                                              values_out);
                else
                  eval_val.template hessians<dim - 1, false, false>(values_in,
                                                                    values_in);
              }
            if (next_dim < dim)
              for (unsigned int q = 0; q < np_1; ++q)
                FEEvaluationImplBasisChange<variant,
                                            next_dim,
                                            basis_size_1,
                                            basis_size_2,
                                            1,
                                            Number,
                                            Number2>::
                  do_backward_hessians(
                    transformation_matrix,
                    add_into_result,
                    values_in +
                      q * dealii::Utilities::fixed_power<next_dim>(np_2),
                    values_out +
                      q * dealii::Utilities::fixed_power<next_dim>(np_1),
                    basis_size_1_variable,
                    basis_size_2_variable);

            values_in += dealii::Utilities::fixed_power<dim>(np_2);
            values_out += dealii::Utilities::fixed_power<dim>(np_1);
          }
      }
    };

  } // namespace internal

} // namespace hyperdeal


#endif
