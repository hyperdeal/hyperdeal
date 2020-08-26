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

#ifndef HYPERDEAL_NDIM_OPERATORS_ADVECTIONOPERATION
#define HYPERDEAL_NDIM_OPERATORS_ADVECTIONOPERATION

#include <hyper.deal/base/config.h>

#include <hyper.deal/base/dynamic_convergence_table.h>
#include <hyper.deal/matrix_free/fe_evaluation_cell.h>
#include <hyper.deal/matrix_free/fe_evaluation_cell_inverse.h>
#include <hyper.deal/matrix_free/fe_evaluation_face.h>
#include <hyper.deal/matrix_free/matrix_free.h>
#include <hyper.deal/matrix_free/tools.h>
#include <hyper.deal/operators/advection/boundary_descriptor.h>

namespace hyperdeal
{
  namespace advection
  {
    /**
     * Interpolate values at quadrature point to faces and transposed
     * operation.
     */
    template <unsigned int DIM,
              unsigned int points,
              unsigned int d,
              bool         forward,
              int          stride,
              typename Number>
    void
    interpolate_to_face(Number *      output,
                        const Number *input,
                        const Number *weights)
    {
      for (auto i = 0u, e = 0u; i < dealii::Utilities::pow(points, DIM - d - 1);
           i++)
        for (auto j = 0u; j < dealii::Utilities::pow(points, d); j++, e++)
          {
            if (forward)
              output[e] = 0;

            for (auto k = 0u; k < points; k++)
              if (forward)
                output[e] += input[i * dealii::Utilities::pow(points, d + 1) +
                                   k * dealii::Utilities::pow(points, d) + j] *
                             weights[k * stride];
              else
                output[i * dealii::Utilities::pow(points, d + 1) +
                       k * dealii::Utilities::pow(points, d) + j] +=
                  input[e] * weights[k * stride];
          }
    }



    /**
     * Advection operator. It is defined by a velocity field and by boundary
     * conditions.
     */
    template <int dim_x,
              int dim_v,
              int degree,
              int n_points,
              typename Number,
              typename VectorType,
              typename VelocityField,
              typename VectorizedArrayType>
    class AdvectionOperation
    {
    public:
      using This    = AdvectionOperation<dim_x,
                                      dim_v,
                                      degree,
                                      n_points,
                                      Number,
                                      VectorType,
                                      VelocityField,
                                      VectorizedArrayType>;
      using VNumber = VectorizedArrayType;

      static const int dim = dim_x + dim_v;

      static const dealii::internal::EvaluatorVariant tensorproduct =
        dealii::internal::EvaluatorVariant::evaluate_evenodd;

      using FECellEval =
        FEEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>;
      using FEFaceEval =
        FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>;
      using FECellEval_inv =
        FEEvaluationInverse<dim_x, dim_v, degree, Number, VNumber>;


      /**
       * Constructor
       */
      AdvectionOperation(
        const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &data,
        DynamicConvergenceTable &                                    table)
        : data(data)
        , table(table)
        , shi_get(dealii::QGaussLobatto<1>(n_points),
                  dealii::FE_DGQArbitraryNodes<1>(dealii::QGauss<1>(n_points)))
      {}

      /**
       * Set boundary condition and velocity field as well as set up internal
       * data structures.
       */
      void
      reinit(
        std::shared_ptr<BoundaryDescriptor<dim, Number>> boundary_descriptor,
        std::shared_ptr<VelocityField>                   velocity_field)
      {
        this->boundary_descriptor = boundary_descriptor;
        this->velocity_field      = velocity_field;

        // clang-format off
        phi_cell.reset(new FEEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>(data, 0, 0, 0, 0));
        phi_cell_inv.reset(new FEEvaluationInverse<dim_x, dim_v, degree, Number, VNumber>(data,0,0, 1, 1));
        phi_face_m.reset(new FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>(data, true, 0, 0, 0, 0));
        phi_face_p.reset(new FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>(data, false, 0, 0, 0, 0));
        // clang-format on
      }

      /**
       * Apply operator. Depending on configuration ECL or FCL.
       */
      void
      apply(VectorType &      dst,
            const VectorType &src,
            const Number      time,
            Timers *          timers = nullptr)
      {
        // set time of boundary functions
        boundary_descriptor->set_time(time);

        // TODO: also for velocity field

        // loop over all cells/faces in phase-space
        if (!data.is_ecl_supported()) // FCL
          {
            if (timers != nullptr)
              timers->enter("FCL");

            // advection operator
            {
              hyperdeal::ScopedTimerWrapper timer(timers, "advection");
              if (timers != nullptr)
                timers->enter("advection");

              data.loop(&This::local_apply_cell,
                        &This::local_apply_face,
                        &This::local_apply_boundary,
                        this,
                        dst,
                        src,
                        MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
                          DataAccessOnFaces::values,
                        MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
                          DataAccessOnFaces::values,
                        timers);

              if (timers != nullptr)
                timers->leave();
            }

            // inverse-mass matrix operator
            {
              hyperdeal::ScopedTimerWrapper timer(timers, "mass");

              data.cell_loop(&This::local_apply_inverse_mass_matrix,
                             this,
                             dst,
                             dst);
            }

            if (timers != nullptr)
              timers->leave();
          }
        else // ECL
          {
            if (timers != nullptr)
              timers->enter("ECL");

            // advection and inverse-mass matrix operator in one go
            data.loop_cell_centric(
              &This::local_apply_advect_and_inverse_mass_matrix,
              this,
              dst,
              src,
              MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
                DataAccessOnFaces::values,
              timers);

            if (timers != nullptr)
              timers->leave();
          }
      }

    private:
      using ID =
        typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::ID;

      /**
       * Advection + inverse mass-matrix cell operation -> ECL.
       */
      void
      local_apply_advect_and_inverse_mass_matrix(
        const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &data,
        VectorType &                                                 dst,
        const VectorType &                                           src,
        const ID                                                     cell)
      {
        (void)data;

        auto &phi     = *this->phi_cell;
        auto &phi_m   = *this->phi_face_m;
        auto &phi_p   = *this->phi_face_p;
        auto &phi_inv = *this->phi_cell_inv;

        // get data and scratch
        VNumber *data_ptr     = phi.get_data_ptr();
        VNumber *data_ptr1    = phi_m.get_data_ptr();
        VNumber *data_ptr2    = phi_p.get_data_ptr();
        VNumber *data_ptr_inv = phi_inv.get_data_ptr();

        // initialize tensor product kernels
        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber>
          eval(*phi.get_shape_values(),
               *phi.get_shape_gradients(),
               *phi.get_shape_gradients() /*DUMMY VALUE -> TODO*/);

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim - 1,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber>
          eval_face(*phi.get_shape_values(),
                    *phi.get_shape_gradients(),
                    *phi.get_shape_gradients() /*DUMMY VALUE -> TODO*/);

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       n_points,
                                                       n_points,
                                                       VNumber>
          eval_(*phi.get_shape_values(),
                *phi.get_shape_gradients(),
                *phi.get_shape_gradients() /*DUMMY VALUE -> TODO*/);

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       degree + 1,
                                                       degree + 1,
                                                       VNumber>
          eval_inv(*phi_inv.get_inverse_shape(),
                   *phi_inv.get_inverse_shape(),
                   *phi_inv.get_inverse_shape() /*DUMMY VALUE -> TODO*/);

        // clang-format off

        // 1) advection: cell contribution
        {
          this->velocity_field->reinit(cell);

          // load from global structure
          phi.reinit(cell);
          phi.read_dof_values(src);

#ifndef COLLOCATION
          if (dim >= 1) eval.template values<0, true, false>(data_ptr, data_ptr);
          if (dim >= 2) eval.template values<1, true, false>(data_ptr, data_ptr);
          if (dim >= 3) eval.template values<2, true, false>(data_ptr, data_ptr);
          if (dim >= 4) eval.template values<3, true, false>(data_ptr, data_ptr);
          if (dim >= 5) eval.template values<4, true, false>(data_ptr, data_ptr);
          if (dim >= 6) eval.template values<5, true, false>(data_ptr, data_ptr);
#endif

          // copy quadrature values into buffer
          VNumber *buffer = phi_cell_inv->get_data_ptr();
          for (auto i = 0u; i < dealii::Utilities::pow<unsigned int>(n_points, dim); i++)
            buffer[i] = data_ptr[i];

          // x-space
          {
            dealii::AlignedVector<VNumber> scratch_data_array;
            scratch_data_array.resize_fast(dealii::Utilities::pow(n_points, dim) * dim_x);
            VNumber *tempp = scratch_data_array.begin();
            for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
              for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                {
                  VNumber    grad_in[dim_x];
                  const auto vel = velocity_field->evaluate_x(q, qx, qv);
                  for (int d = 0; d < dim_x; d++)
                    grad_in[d] = buffer[q] * vel[d];
                  phi.submit_gradient_x(tempp, grad_in, q, qx, qv);
                }
            if (dim_x >= 1) eval_.template gradients<0, false, false>(tempp + dealii::Utilities::pow(n_points, dim) * 0, data_ptr);
            if (dim_x >= 2) eval_.template gradients<1, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 1, data_ptr);
            if (dim_x >= 3) eval_.template gradients<2, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 2, data_ptr);
          }
          // v-space
          {
            dealii::AlignedVector<VNumber> scratch_data_array;
            scratch_data_array.resize_fast(dealii::Utilities::pow(n_points, dim) * dim_v);
            VNumber *tempp = scratch_data_array.begin();
            for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
              for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                {
                  VNumber    grad_in[dim_v];
                  const auto vel = velocity_field->evaluate_v(q, qx, qv);
                  for (int d = 0; d < dim_v; d++)
                    grad_in[d] = buffer[q] * vel[d];
                  phi.submit_gradient_v(tempp, grad_in, q, qx, qv);
                }

            if (dim_v >= 1) eval_.template gradients<0 + dim_x, false, true>(tempp + dealii::Utilities::pow(n_points, dim) * 0, data_ptr);
            if (dim_v >= 2) eval_.template gradients<1 + dim_x, false, true>(tempp + dealii::Utilities::pow(n_points, dim) * 1, data_ptr);
            if (dim_v >= 3) eval_.template gradients<2 + dim_x, false, true>(tempp + dealii::Utilities::pow(n_points, dim) * 2, data_ptr);
          }
        }

        // 2) advection: faces
        for (auto face = 0u; face < dim * 2; face++)
          {
            this->velocity_field->reinit_face(cell, face);

            // load negative side from buffer
            phi_m.reinit(cell, face);
            
            const auto bid = data.get_faces_by_cells_boundary_id(cell, face);
            
            // load positive side from global structure
            if(bid == dealii::numbers::internal_face_boundary_id)
            {
              phi_p.reinit(cell, face);
              phi_p.read_dof_values(src);
            }

#ifndef COLLOCATION
            const auto weights = &shi_get.data[0].shape_values[face % 2 == 0 ? 0 : (n_points - 1)];
            if (dim >= 1 && face / 2 == 0) interpolate_to_face<dim, n_points, 0, true, n_points>(data_ptr1, data_ptr_inv, weights); else
            if (dim >= 2 && face / 2 == 1) interpolate_to_face<dim, n_points, 1, true, n_points>(data_ptr1, data_ptr_inv, weights); else
            if (dim >= 3 && face / 2 == 2) interpolate_to_face<dim, n_points, 2, true, n_points>(data_ptr1, data_ptr_inv, weights); else
            if (dim >= 4 && face / 2 == 3) interpolate_to_face<dim, n_points, 3, true, n_points>(data_ptr1, data_ptr_inv, weights); else
            if (dim >= 5 && face / 2 == 4) interpolate_to_face<dim, n_points, 4, true, n_points>(data_ptr1, data_ptr_inv, weights); else
            if (dim >= 6 && face / 2 == 5) interpolate_to_face<dim, n_points, 5, true, n_points>(data_ptr1, data_ptr_inv, weights);

            if (dim >= 2) eval_face.template values<0, true, false>(data_ptr2, data_ptr2);
            if (dim >= 3) eval_face.template values<1, true, false>(data_ptr2, data_ptr2);
            if (dim >= 4) eval_face.template values<2, true, false>(data_ptr2, data_ptr2);
            if (dim >= 5) eval_face.template values<3, true, false>(data_ptr2, data_ptr2);
            if (dim >= 6) eval_face.template values<4, true, false>(data_ptr2, data_ptr2);
#else
            phi_m.read_dof_values_from_buffer(this->phi_cell_inv->get_data_ptr());
#endif

            if(bid == dealii::numbers::internal_face_boundary_id)
              {
                if (face < dim_x * 2)
                  {
                    for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
                      for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x - 1); ++qx, ++q)
                        {
                          const VectorizedArrayType u_minus                = data_ptr1[q];
                          const VectorizedArrayType u_plus                 = data_ptr2[q];
                          const VectorizedArrayType normal_times_advection = velocity_field->evaluate_face_x(q, qx, qv) * phi_m.template get_normal_vector_x(qx);
                          const VectorizedArrayType flux_times_normal      = 0.5 * ((u_minus + u_plus) * normal_times_advection + std::abs(normal_times_advection) * (u_minus - u_plus)) * alpha;
        
                          phi_m.template submit_value<ID::SpaceType::X>(data_ptr1, flux_times_normal, q, qx, qv);
                        }
                  }
                else
                  {
                    for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v - 1); ++qv)
                      for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                        {
                          const VectorizedArrayType u_minus                = data_ptr1[q];
                          const VectorizedArrayType u_plus                 = data_ptr2[q];
                          const VectorizedArrayType normal_times_advection = velocity_field->evaluate_face_v(q, qx, qv) * phi_m.template get_normal_vector_v(qv);
                          const VectorizedArrayType flux_times_normal      = 0.5 * ((u_minus + u_plus) * normal_times_advection + std::abs(normal_times_advection) * (u_minus - u_plus)) * alpha;
        
                          phi_m.template submit_value<ID::SpaceType::V>(data_ptr1, flux_times_normal, q, qx, qv);
                        }
                  }
              }
            else
              {
                const auto boundary_pair = boundary_descriptor->get_boundary(bid);
                    
                if (face < dim_x * 2)
                  {
                    for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
                      for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x - 1); ++qx, ++q)
                        {
                          const VectorizedArrayType u_minus = data_ptr1[q];
                          const VectorizedArrayType u_plus = boundary_pair.first == BoundaryType::DirichletHomogenous ? 
                              (-u_minus) : 
                              (-u_minus + 2.0 * dealii::MatrixFreeTools::evaluate_scalar_function(phi_m.template get_quadrature_point<ID::SpaceType::X>(qx, qv), *boundary_pair.second, phi_m.n_vectorization_lanes_filled()));
                          
                          const VectorizedArrayType normal_times_advection = velocity_field->evaluate_face_x(q, qx, qv) * phi_m.template get_normal_vector_x(qx);
                          const VectorizedArrayType flux_times_normal      = 0.5 * ((u_minus + u_plus) * normal_times_advection + std::abs(normal_times_advection) * (u_minus - u_plus)) * alpha;
        
                          phi_m.template submit_value<ID::SpaceType::X>(data_ptr1, flux_times_normal, q, qx, qv);
                        }
                  }
                else
                  {
                    for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v - 1); ++qv)
                      for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                        {
                          const VectorizedArrayType u_minus = data_ptr1[q];
                          const VectorizedArrayType u_plus = boundary_pair.first == BoundaryType::DirichletHomogenous ? 
                              (-u_minus) : 
                              (-u_minus + 2.0 * dealii::MatrixFreeTools::evaluate_scalar_function(phi_m.template get_quadrature_point<ID::SpaceType::V>(qx, qv), *boundary_pair.second, phi_m.n_vectorization_lanes_filled()));
                          
                          const VectorizedArrayType normal_times_advection = velocity_field->evaluate_face_v(q, qx, qv) * phi_m.template get_normal_vector_v(qv);
                          const VectorizedArrayType flux_times_normal      = 0.5 * ((u_minus + u_plus) * normal_times_advection + std::abs(normal_times_advection) * (u_minus - u_plus)) * alpha;
        
                          phi_m.template submit_value<ID::SpaceType::V>(data_ptr1, flux_times_normal, q, qx, qv);
                        }
                  }
              }

#ifndef COLLOCATION
            {
              const auto weights = &shi_get.data[0].shape_values[face % 2 == 0 ? 0 : (n_points - 1)];
              if (dim >= 1 && face / 2 == 0) interpolate_to_face<dim, n_points, 0, false, n_points>(data_ptr, data_ptr1, weights); else
              if (dim >= 2 && face / 2 == 1) interpolate_to_face<dim, n_points, 1, false, n_points>(data_ptr, data_ptr1, weights); else
              if (dim >= 3 && face / 2 == 2) interpolate_to_face<dim, n_points, 2, false, n_points>(data_ptr, data_ptr1, weights); else
              if (dim >= 4 && face / 2 == 3) interpolate_to_face<dim, n_points, 3, false, n_points>(data_ptr, data_ptr1, weights); else
              if (dim >= 5 && face / 2 == 4) interpolate_to_face<dim, n_points, 4, false, n_points>(data_ptr, data_ptr1, weights); else
              if (dim >= 6 && face / 2 == 5) interpolate_to_face<dim, n_points, 5, false, n_points>(data_ptr, data_ptr1, weights);
            }
#else
            phi_m.distribute_to_buffer(this->phi_cell->get_data_ptr());
#endif
          }

        // 3) inverse mass matrix
        {
          phi_inv.reinit(cell);

          for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
            for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
              phi_inv.submit_inv(data_ptr, q, qx, qv);

#ifndef COLLOCATION
          if (dim >= 6) eval_inv.template hessians<5, false, false>(data_ptr, data_ptr);
          if (dim >= 5) eval_inv.template hessians<4, false, false>(data_ptr, data_ptr);
          if (dim >= 4) eval_inv.template hessians<3, false, false>(data_ptr, data_ptr);
          if (dim >= 3) eval_inv.template hessians<2, false, false>(data_ptr, data_ptr);
          if (dim >= 2) eval_inv.template hessians<1, false, false>(data_ptr, data_ptr);
          if (dim >= 1) eval_inv.template hessians<0, false, false>(data_ptr, data_ptr);
#endif

          // write into global structure back
          phi.set_dof_values(dst);
        }

        // clang-format on
      }



      /**
       * Inverse mass-matrix cell operation -> FCL.
       */
      void
      local_apply_inverse_mass_matrix(
        const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &data,
        VectorType &                                                 dst,
        const VectorType &                                           src,
        const ID                                                     cell)
      {
        (void)data;

        auto &phi_inv = *this->phi_cell_inv;

        // get data and scratch
        VectorizedArrayType *data_ptr = phi_inv.get_data_ptr();

        // load from global structure
        phi_inv.reinit(cell);
        phi_inv.read_dof_values(src);

        // initialize tensor product kernels
        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       degree + 1,
                                                       degree + 1,
                                                       VectorizedArrayType>
          eval_inv(*phi_inv.get_inverse_shape(),
                   *phi_inv.get_inverse_shape(),
                   *phi_inv.get_inverse_shape() /*DUMMY VALUE -> TODO*/);

        // clang-format off
        
#ifndef COLLOCATION
        if (dim >= 1) eval_inv.template hessians<0, true, false>(data_ptr, data_ptr);
        if (dim >= 2) eval_inv.template hessians<1, true, false>(data_ptr, data_ptr);
        if (dim >= 3) eval_inv.template hessians<2, true, false>(data_ptr, data_ptr);
        if (dim >= 4) eval_inv.template hessians<3, true, false>(data_ptr, data_ptr);
        if (dim >= 5) eval_inv.template hessians<4, true, false>(data_ptr, data_ptr);
        if (dim >= 6) eval_inv.template hessians<5, true, false>(data_ptr, data_ptr);
#endif

        for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(degree + 1, dim_v); ++qv)
          for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(degree + 1, dim_x); ++qx, ++q)
            phi_inv.submit_inv(data_ptr, q, qx, qv);

#ifndef COLLOCATION
        if (dim >= 6) eval_inv.template hessians<5, false, false>(data_ptr, data_ptr);
        if (dim >= 5) eval_inv.template hessians<4, false, false>(data_ptr, data_ptr);
        if (dim >= 4) eval_inv.template hessians<3, false, false>(data_ptr, data_ptr);
        if (dim >= 3) eval_inv.template hessians<2, false, false>(data_ptr, data_ptr);
        if (dim >= 2) eval_inv.template hessians<1, false, false>(data_ptr, data_ptr);
        if (dim >= 1) eval_inv.template hessians<0, false, false>(data_ptr, data_ptr);
#endif

        // clang-format on

        // write into global structure back
        phi_inv.set_dof_values(dst);
      }



      /**
       * Advection cell operation -> FCL.
       */
      void
      local_apply_cell(
        const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &data,
        VectorType &                                                 dst,
        const VectorType &                                           src,
        const ID                                                     cell)
      {
        (void)data;

        auto &phi = *this->phi_cell;

        // get data and scratch
        VNumber *data_ptr = phi.get_data_ptr();

        // initialize tensor product kernels
        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber>
          eval(*phi.get_shape_values(),
               *phi.get_shape_gradients(),
               *phi.get_shape_gradients() /*DUMMY VALUE -> TODO*/);

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       n_points,
                                                       n_points,
                                                       VNumber>
          eval_(*phi.get_shape_values(),
                *phi.get_shape_gradients(),
                *phi.get_shape_gradients() /*DUMMY VALUE -> TODO*/);

        // clang-format off

        this->velocity_field->reinit(cell);

        // load from global structure
        phi.reinit(cell);
        phi.read_dof_values(src);

#ifndef COLLOCATION
        if (dim >= 1) eval.template values<0, true, false>(data_ptr, data_ptr);
        if (dim >= 2) eval.template values<1, true, false>(data_ptr, data_ptr);
        if (dim >= 3) eval.template values<2, true, false>(data_ptr, data_ptr);
        if (dim >= 4) eval.template values<3, true, false>(data_ptr, data_ptr);
        if (dim >= 5) eval.template values<4, true, false>(data_ptr, data_ptr);
        if (dim >= 6) eval.template values<5, true, false>(data_ptr, data_ptr);
#endif

        // copy quadrature values into buffer
        VNumber *buffer = phi_cell_inv->get_data_ptr();
        for (auto i = 0u; i < dealii::Utilities::pow<unsigned int>(n_points, dim); i++)
          buffer[i] = data_ptr[i];

        // x-space
        {
          dealii::AlignedVector<VNumber> scratch_data_array;
          scratch_data_array.resize_fast(dealii::Utilities::pow(n_points, dim) * dim_x);
          VNumber *tempp = scratch_data_array.begin();
          for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
            for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
              {
                VNumber    grad_in[dim_x];
                const auto vel = velocity_field->evaluate_x(q, qx, qv);
                for (int d = 0; d < dim_x; d++)
                  grad_in[d] = buffer[q] * vel[d];
                phi.submit_gradient_x(tempp, grad_in, q, qx, qv);
              }
          if (dim_x >= 1) eval_.template gradients<0, false, false>(tempp + dealii::Utilities::pow(n_points, dim) * 0, data_ptr);
          if (dim_x >= 2) eval_.template gradients<1, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 1, data_ptr);
          if (dim_x >= 3) eval_.template gradients<2, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 2, data_ptr);
        }
        // v-space
        {
          dealii::AlignedVector<VNumber> scratch_data_array;
          scratch_data_array.resize_fast(dealii::Utilities::pow(n_points, dim) * dim_v);
          VNumber *tempp = scratch_data_array.begin();
          for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
            for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
              {
                VNumber    grad_in[dim_v];
                const auto vel = velocity_field->evaluate_v(q, qx, qv);
                for (int d = 0; d < dim_v; d++)
                  grad_in[d] = buffer[q] * vel[d];
                phi.submit_gradient_v(tempp, grad_in, q, qx, qv);
              }

          if (dim_v >= 1) eval_.template gradients<0 + dim_x, false, true>(tempp + dealii::Utilities::pow(n_points, dim) * 0, data_ptr);
          if (dim_v >= 2) eval_.template gradients<1 + dim_x, false, true>(tempp + dealii::Utilities::pow(n_points, dim) * 1, data_ptr);
          if (dim_v >= 3) eval_.template gradients<2 + dim_x, false, true>(tempp + dealii::Utilities::pow(n_points, dim) * 2, data_ptr);
        }

#ifndef COLLOCATION
        if(dim >= 6) eval.template values<5, false, false>(data_ptr, data_ptr);
        if(dim >= 5) eval.template values<4, false, false>(data_ptr, data_ptr);
        if(dim >= 4) eval.template values<3, false, false>(data_ptr, data_ptr);
        if(dim >= 3) eval.template values<2, false, false>(data_ptr, data_ptr);
        if(dim >= 2) eval.template values<1, false, false>(data_ptr, data_ptr);
        if(dim >= 1) eval.template values<0, false, false>(data_ptr, data_ptr);
#endif

        // clang-format on

        // write into global structure back
        phi.set_dof_values(dst);
      }



      /**
       * Advection face operation -> FCL.
       */
      void
      local_apply_face(
        const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &data,
        VectorType &                                                 dst,
        const VectorType &                                           src,
        const ID                                                     face)
      {
        (void)data;

        auto &phi_m = *this->phi_face_m;
        auto &phi_p = *this->phi_face_p;

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim - 1,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber>
          eval1(*phi_m.get_shape_values(),
                *phi_m.get_shape_gradients(),
                *phi_m.get_shape_gradients() /*DUMMY VALUE -> TODO*/);
        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim - 1,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber>
          eval2(*phi_p.get_shape_values(),
                *phi_p.get_shape_gradients(),
                *phi_p.get_shape_gradients() /*DUMMY VALUE -> TODO*/);

        this->velocity_field->reinit_face(face);

        // get data and scratch
        VNumber *data_ptr1 = phi_m.get_data_ptr();
        VNumber *data_ptr2 = phi_p.get_data_ptr();

        // load from global structure
        phi_m.reinit(face);
        phi_p.reinit(face);

        // clang-format off

        phi_m.read_dof_values(src);
#ifndef COLLOCATION
        if (dim >= 2) eval1.template values<0, true, false>(data_ptr1, data_ptr1);
        if (dim >= 3) eval1.template values<1, true, false>(data_ptr1, data_ptr1);
        if (dim >= 4) eval1.template values<2, true, false>(data_ptr1, data_ptr1);
        if (dim >= 5) eval1.template values<3, true, false>(data_ptr1, data_ptr1);
        if (dim >= 6) eval1.template values<4, true, false>(data_ptr1, data_ptr1);
#endif

        phi_p.read_dof_values(src);
#ifndef COLLOCATION
        if (dim >= 2) eval2.template values<0, true, false>(data_ptr2, data_ptr2);
        if (dim >= 3) eval2.template values<1, true, false>(data_ptr2, data_ptr2);
        if (dim >= 4) eval2.template values<2, true, false>(data_ptr2, data_ptr2);
        if (dim >= 5) eval2.template values<3, true, false>(data_ptr2, data_ptr2);
        if (dim >= 6) eval2.template values<4, true, false>(data_ptr2, data_ptr2);
#endif

        if (face.type == ID::SpaceType::X)
          {
            for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
              for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x - 1); ++qx, ++q)
                {
                  const VectorizedArrayType u_minus                = data_ptr1[q];
                  const VectorizedArrayType u_plus                 = data_ptr2[q];
                  const VectorizedArrayType normal_times_advection = velocity_field->evaluate_face_x(q, qx, qv) * phi_m.template get_normal_vector_x(qx);
                  const VectorizedArrayType flux_times_normal      = 0.5 * ((u_minus + u_plus) * normal_times_advection + std::abs(normal_times_advection) * (u_minus - u_plus)) * alpha;

                  phi_m.template submit_value<ID::SpaceType::X>(data_ptr1, data_ptr2, flux_times_normal, q, qx, qv);
                }
          }
        else
          {
            for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v - 1); ++qv)
              for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                {
                  const VectorizedArrayType u_minus                = data_ptr1[q];
                  const VectorizedArrayType u_plus                 = data_ptr2[q];
                  const VectorizedArrayType normal_times_advection = velocity_field->evaluate_face_v(q, qx, qv) * phi_m.template get_normal_vector_v(qv);
                  const VectorizedArrayType flux_times_normal      = 0.5 * ((u_minus + u_plus) * normal_times_advection + std::abs(normal_times_advection) * (u_minus - u_plus)) * alpha;

                  phi_m.template submit_value<ID::SpaceType::V>(data_ptr1, data_ptr2, flux_times_normal, q, qx, qv);
                }
          }

#ifndef COLLOCATION
        if (dim >= 6) eval1.template values<4, false, false>(data_ptr1, data_ptr1);
        if (dim >= 5) eval1.template values<3, false, false>(data_ptr1, data_ptr1);
        if (dim >= 4) eval1.template values<2, false, false>(data_ptr1, data_ptr1);
        if (dim >= 3) eval1.template values<1, false, false>(data_ptr1, data_ptr1);
        if (dim >= 2) eval1.template values<0, false, false>(data_ptr1, data_ptr1);
#endif

        // write into global structure back
        phi_m.distribute_local_to_global(dst);

#ifndef COLLOCATION
        if (dim >= 6) eval2.template values<4, false, false>(data_ptr2, data_ptr2);
        if (dim >= 5) eval2.template values<3, false, false>(data_ptr2, data_ptr2);
        if (dim >= 4) eval2.template values<2, false, false>(data_ptr2, data_ptr2);
        if (dim >= 3) eval2.template values<1, false, false>(data_ptr2, data_ptr2);
        if (dim >= 2) eval2.template values<0, false, false>(data_ptr2, data_ptr2);
#endif

        // clang-format on

        // write into global structure back
        phi_p.distribute_local_to_global(dst);
      }



      /**
       * Advection boundary operation -> FCL.
       *
       * @note Not implemented yet (TODO).
       */
      void
      local_apply_boundary(
        const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &data,
        VectorType &                                                 dst,
        const VectorType &                                           src,
        const ID                                                     face)
      {
        (void)data;

        const auto bid = data.get_boundary_id(face);

        Assert(bid != dealii::numbers::internal_face_boundary_id,
               dealii::StandardExceptions::ExcInternalError());

        const auto boundary_pair = boundary_descriptor->get_boundary(bid);

        auto &phi_m = *this->phi_face_m;

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim - 1,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber>
          eval1(*phi_m.get_shape_values(),
                *phi_m.get_shape_gradients(),
                *phi_m.get_shape_gradients() /*DUMMY VALUE -> TODO*/);

        this->velocity_field->reinit_face(face);

        // get data and scratch
        VNumber *data_ptr1 = phi_m.get_data_ptr();

        // load from global structure
        phi_m.reinit(face);

        // clang-format off

        phi_m.read_dof_values(src);
#ifndef COLLOCATION
        if (dim >= 2) eval1.template values<0, true, false>(data_ptr1, data_ptr1);
        if (dim >= 3) eval1.template values<1, true, false>(data_ptr1, data_ptr1);
        if (dim >= 4) eval1.template values<2, true, false>(data_ptr1, data_ptr1);
        if (dim >= 5) eval1.template values<3, true, false>(data_ptr1, data_ptr1);
        if (dim >= 6) eval1.template values<4, true, false>(data_ptr1, data_ptr1);
#endif

        if (face.type == ID::SpaceType::X)
          {
            for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
              for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x - 1); ++qx, ++q)
                {
                  const VectorizedArrayType u_minus = data_ptr1[q];
                  const VectorizedArrayType u_plus = (boundary_pair.first == BoundaryType::DirichletHomogenous) ? 
                      (-u_minus) : 
                      (-u_minus + 2.0 * dealii::MatrixFreeTools::evaluate_scalar_function(phi_m.template get_quadrature_point<ID::SpaceType::X>(qx, qv), *boundary_pair.second, phi_m.n_vectorization_lanes_filled()));
                  
                  const VectorizedArrayType normal_times_advection = velocity_field->evaluate_face_x(q, qx, qv) * phi_m.template get_normal_vector_x(qx);
                  const VectorizedArrayType flux_times_normal      = 0.5 * ((u_minus + u_plus) * normal_times_advection + std::abs(normal_times_advection) * (u_minus - u_plus)) * alpha;

                  phi_m.template submit_value<ID::SpaceType::X>(data_ptr1, flux_times_normal, q, qx, qv);
                }
          }
        else
          {
            for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v - 1); ++qv)
              for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                {
                  const VectorizedArrayType u_minus = data_ptr1[q];
                  const VectorizedArrayType u_plus = (boundary_pair.first == BoundaryType::DirichletHomogenous) ? 
                      (-u_minus) : 
                      (-u_minus + 2.0 * dealii::MatrixFreeTools::evaluate_scalar_function(phi_m.template get_quadrature_point<ID::SpaceType::V>(qx, qv), *boundary_pair.second, phi_m.n_vectorization_lanes_filled()));
                  
                  const VectorizedArrayType normal_times_advection = velocity_field->evaluate_face_v(q, qx, qv) * phi_m.template get_normal_vector_v(qv);
                  const VectorizedArrayType flux_times_normal      = 0.5 * ((u_minus + u_plus) * normal_times_advection + std::abs(normal_times_advection) * (u_minus - u_plus)) * alpha;

                  phi_m.template submit_value<ID::SpaceType::V>(data_ptr1, flux_times_normal, q, qx, qv);
                }
          }

#ifndef COLLOCATION
        if (dim >= 6) eval1.template values<4, false, false>(data_ptr1, data_ptr1);
        if (dim >= 5) eval1.template values<3, false, false>(data_ptr1, data_ptr1);
        if (dim >= 4) eval1.template values<2, false, false>(data_ptr1, data_ptr1);
        if (dim >= 3) eval1.template values<1, false, false>(data_ptr1, data_ptr1);
        if (dim >= 2) eval1.template values<0, false, false>(data_ptr1, data_ptr1);
#endif

        // write into global structure back
        phi_m.distribute_local_to_global(dst);
      }

      const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &data;
      DynamicConvergenceTable &                                    table;

      dealii::internal::MatrixFreeFunctions::ShapeInfo<VectorizedArrayType>
        shi_get;

      // clang-format off
      std::shared_ptr<FEEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>> phi_cell;
      std::shared_ptr<FEEvaluationInverse<dim_x, dim_v, degree, Number, VNumber>> phi_cell_inv;
      std::shared_ptr<FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>> phi_face_m;
      std::shared_ptr<FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>> phi_face_p;
      // clang-format on

      std::shared_ptr<BoundaryDescriptor<dim, Number>> boundary_descriptor;
      std::shared_ptr<VelocityField>                   velocity_field;

      const double alpha = 1.0;
    };
  } // namespace advection
} // namespace hyperdeal


#endif
