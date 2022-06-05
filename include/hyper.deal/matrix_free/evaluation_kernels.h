#ifndef HYPERDEAL_NDIM_MATRIXFREE_EVALUTION_KERNELS
#define HYPERDEAL_NDIM_MATRIXFREE_EVALUTION_KERNELS

#include <hyper.deal/base/config.h>

#include <deal.II/matrix_free/evaluation_flags.h>

namespace hyperdeal
{
  namespace internal
  {
    template <int dim_x,
              int dim_v,
              int n_rows,
              int n_columns,
              typename Number,
              typename Number2 = Number>
    struct EvaluatorTensorProduct
    {
      static constexpr int          dim = dim_x + dim_v;
      static constexpr unsigned int n_rows_of_product =
        dealii::Utilities::pow(n_rows, dim);
      static constexpr unsigned int n_columns_of_product =
        dealii::Utilities::pow(n_columns, dim);

      EvaluatorTensorProduct()
        : shape_values(nullptr)
        , shape_gradients(nullptr)
        , shape_hessians(nullptr)
      {}

      EvaluatorTensorProduct(
        const dealii::AlignedVector<Number2> &shape_values,
        const dealii::AlignedVector<Number2> &shape_gradients,
        const dealii::AlignedVector<Number2> &shape_hessians,
        const unsigned int                    dummy1 = 0,
        const unsigned int                    dummy2 = 0)
        : shape_values(shape_values.begin())
        , shape_gradients(shape_gradients.begin())
        , shape_hessians(shape_hessians.begin())
      {
        // We can enter this function either for the apply() path that has
        // n_rows * n_columns entries or for the apply_face() path that only has
        // n_rows * 3 entries in the array. Since we cannot decide about the use
        // we must allow for both here.
        Assert(shape_values.size() == 0 ||
                 shape_values.size() == n_rows * n_columns ||
                 shape_values.size() == 3 * n_rows,
               dealii::ExcDimensionMismatch(shape_values.size(),
                                            n_rows * n_columns));
        Assert(shape_gradients.size() == 0 ||
                 shape_gradients.size() == n_rows * n_columns,
               dealii::ExcDimensionMismatch(shape_gradients.size(),
                                            n_rows * n_columns));
        Assert(shape_hessians.size() == 0 ||
                 shape_hessians.size() == n_rows * n_columns,
               dealii::ExcDimensionMismatch(shape_hessians.size(),
                                            n_rows * n_columns));
        (void)dummy1;
        (void)dummy2;
      }

      template <int  face_direction,
                bool contract_onto_face,
                bool add,
                int  max_derivative>
      void
      apply_face(const Number *DEAL_II_RESTRICT in,
                 Number *DEAL_II_RESTRICT out) const
      {
        if ((dim_x == 3) && (face_direction == 1))
          {
            for (int i = 0; i < dealii::Utilities::pow(n_rows, dim - 1); ++i)
              out[i] = -1.0;
          }
        else if ((dim_v == 3) && (face_direction == (1 + dim_x)))
          {
            for (int i = 0; i < dealii::Utilities::pow(n_rows, dim - 1); ++i)
              out[i] = -1.0;
          }
        else // lex
          {
            const bool lex_faces = true;

            Assert(dim > 0 && (lex_faces || dim < 4),
                   dealii::ExcMessage("Only dim=1,2,3 supported"));
            static_assert(max_derivative >= 0 && max_derivative < 3,
                          "Only derivative orders 0-2 implemented");
            Assert(
              shape_values != nullptr,
              dealii::ExcMessage(
                "The given array shape_values must not be the null pointer."));

            constexpr int n_blocks1 =
              lex_faces ?
                dealii::Utilities::pow<unsigned int>(n_rows, face_direction) :
                (dim > 1 ? n_rows : 1);
            constexpr int n_blocks2 =
              lex_faces ? dealii::Utilities::pow<unsigned int>(
                            n_rows, std::max(dim - face_direction - 1, 0)) :
                          (dim > 2 ? n_rows : 1);

            AssertIndexRange(face_direction, dim);
            constexpr int stride =
              dealii::Utilities::pow(n_rows, face_direction);
            constexpr int out_stride = dealii::Utilities::pow(n_rows, dim - 1);
            const Number *DEAL_II_RESTRICT shape_values = this->shape_values;

            for (int i2 = 0; i2 < n_blocks2; ++i2)
              {
                for (int i1 = 0; i1 < n_blocks1; ++i1)
                  {
                    if (contract_onto_face == true)
                      {
                        Number res0 = shape_values[0] * in[0];
                        Number res1, res2;
                        if (max_derivative > 0)
                          res1 = shape_values[n_rows] * in[0];
                        if (max_derivative > 1)
                          res2 = shape_values[2 * n_rows] * in[0];
                        for (int ind = 1; ind < n_rows; ++ind)
                          {
                            res0 += shape_values[ind] * in[stride * ind];
                            if (max_derivative > 0)
                              res1 +=
                                shape_values[ind + n_rows] * in[stride * ind];
                            if (max_derivative > 1)
                              res2 += shape_values[ind + 2 * n_rows] *
                                      in[stride * ind];
                          }
                        if (add)
                          {
                            out[0] += res0;
                            if (max_derivative > 0)
                              out[out_stride] += res1;
                            if (max_derivative > 1)
                              out[2 * out_stride] += res2;
                          }
                        else
                          {
                            out[0] = res0;
                            if (max_derivative > 0)
                              out[out_stride] = res1;
                            if (max_derivative > 1)
                              out[2 * out_stride] = res2;
                          }
                      }
                    else
                      {
                        for (int col = 0; col < n_rows; ++col)
                          {
                            if (add)
                              out[col * stride] += shape_values[col] * in[0];
                            else
                              out[col * stride] = shape_values[col] * in[0];
                            if (max_derivative > 0)
                              out[col * stride] +=
                                shape_values[col + n_rows] * in[out_stride];
                            if (max_derivative > 1)
                              out[col * stride] +=
                                shape_values[col + 2 * n_rows] *
                                in[2 * out_stride];
                          }
                      }

                    if (lex_faces)
                      {
                        ++out;
                        ++in;
                      }
                    else
                      // increment: in regular case, just go to the next point
                      // in x-direction. If we are at the end of one chunk in
                      // x-dir, need to jump over to the next layer in
                      // z-direction
                      switch (face_direction)
                        {
                          case 0:
                            in += contract_onto_face ? n_rows : 1;
                            out += contract_onto_face ? 1 : n_rows;
                            break;
                          case 1:
                            ++in;
                            ++out;
                            // faces 2 and 3 in 3D use local coordinate system
                            // zx, which is the other way around compared to the
                            // tensor product. Need to take that into account.
                            if (dim == 3)
                              {
                                if (contract_onto_face)
                                  out += n_rows - 1;
                                else
                                  in += n_rows - 1;
                              }
                            break;
                          case 2:
                            ++in;
                            ++out;
                            break;
                          default:
                            Assert(false, dealii::ExcNotImplemented());
                        }
                  }
                if (lex_faces)
                  {
                    if (contract_onto_face)
                      in +=
                        (dealii::Utilities::pow(n_rows, face_direction + 1) -
                         n_blocks1);
                    else
                      out +=
                        (dealii::Utilities::pow(n_rows, face_direction + 1) -
                         n_blocks1);
                  }
                else if (face_direction == 1 && dim == 3)
                  {
                    // adjust for local coordinate system zx
                    if (contract_onto_face)
                      {
                        in += n_rows * (n_rows - 1);
                        out -= n_rows * n_rows - 1;
                      }
                    else
                      {
                        out += n_rows * (n_rows - 1);
                        in -= n_rows * n_rows - 1;
                      }
                  }
              }
          }
      }

    private:
      const Number2 *shape_values;
      const Number2 *shape_gradients;
      const Number2 *shape_hessians;
    };

    template <int dim_x, int dim_v, int fe_degree, typename Number>
    struct FEFaceNormalEvaluationImpl
    {
      static const int dim = dim_x + dim_v;
      /**
       * Interpolate the values on the cell quadrature points onto a face.
       */
      template <bool do_evaluate, bool add_into_output>
      static void
      interpolate_quadrature(
        const unsigned int                             n_components,
        const dealii::EvaluationFlags::EvaluationFlags flags,
        const dealii::internal::MatrixFreeFunctions::ShapeInfo<Number>
          &                shape_info,
        const Number *     input,
        Number *           output,
        const unsigned int face_no)
      {
        Assert(static_cast<unsigned int>(fe_degree + 1) ==
                   shape_info.data.front().n_q_points_1d ||
                 fe_degree == -1,
               dealii::ExcInternalError());

        interpolate_generic<do_evaluate, add_into_output>(
          n_components,
          input,
          output,
          flags,
          face_no,
          shape_info.data.front().quadrature.size(),
          shape_info.data.front().quadrature_data_on_face,
          shape_info.n_q_points,
          shape_info.n_q_points_face);
      }

    private:
      template <bool do_evaluate, bool add_into_output, int face_direction = 0>
      static void
      interpolate_generic(
        const unsigned int                                  n_components,
        const Number *                                      input,
        Number *                                            output,
        const dealii::EvaluationFlags::EvaluationFlags      flag,
        const unsigned int                                  face_no,
        const unsigned int                                  n_points_1d,
        const std::array<dealii::AlignedVector<Number>, 2> &shape_data,
        const unsigned int dofs_per_component_on_cell,
        const unsigned int dofs_per_component_on_face)
      {
        if (face_direction == face_no / 2)
          {
            EvaluatorTensorProduct<dim_x, dim_v, fe_degree + 1, 0, Number>
              evalf(shape_data[face_no % 2],
                    dealii::AlignedVector<Number>(),
                    dealii::AlignedVector<Number>(),
                    n_points_1d,
                    0);

            const unsigned int in_stride = do_evaluate ?
                                             dofs_per_component_on_cell :
                                             dofs_per_component_on_face;
            const unsigned int out_stride = do_evaluate ?
                                              dofs_per_component_on_face :
                                              dofs_per_component_on_cell;

            for (unsigned int c = 0; c < n_components; ++c)
              {
                if (flag & dealii::EvaluationFlags::hessians)
                  evalf.template apply_face<face_direction,
                                            do_evaluate,
                                            add_into_output,
                                            2>(input, output);
                else if (flag & dealii::EvaluationFlags::gradients)
                  evalf.template apply_face<face_direction,
                                            do_evaluate,
                                            add_into_output,
                                            1>(input, output);
                else
                  evalf.template apply_face<face_direction,
                                            do_evaluate,
                                            add_into_output,
                                            0>(input, output);
                input += in_stride;
                output += out_stride;
              }
          }
        else if (face_direction < dim)
          {
            interpolate_generic<do_evaluate,
                                add_into_output,
                                std::min(face_direction + 1, dim - 1)>(
              n_components,
              input,
              output,
              flag,
              face_no,
              n_points_1d,
              shape_data,
              dofs_per_component_on_cell,
              dofs_per_component_on_face);
          }
      }
    };

  } // namespace internal
} // namespace hyperdeal

#endif
