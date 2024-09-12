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

#include <deal.II/matrix_free/evaluation_kernels.h>

#include <hyper.deal/base/dynamic_convergence_table.h>
#include <hyper.deal/matrix_free/evaluation_kernels.h>
#include <hyper.deal/matrix_free/fe_evaluation_cell.h>
#include <hyper.deal/matrix_free/fe_evaluation_cell_inverse.h>
#include <hyper.deal/matrix_free/fe_evaluation_face.h>
#include <hyper.deal/matrix_free/matrix_free.h>
#include <hyper.deal/matrix_free/tools.h>
#include <hyper.deal/operators/advection/advection_operation_parameters.h>
#include <hyper.deal/operators/advection/boundary_descriptor.h>

namespace hyperdeal
{
  namespace advection
  {
    enum class AdvectionOperationEvaluationLevel
    {
      cell,
      all_without_neighbor_load,
      all
    };

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
        FEEvaluationInverse<dim_x, dim_v, degree, n_points, Number, VNumber>;


      /**
       * Constructor
       */
      AdvectionOperation(
        const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &data,
        DynamicConvergenceTable &                                    table)
        : data(data)
        , table(table)
        , do_collocation(false)
      {}

      /**
       * Set boundary condition and velocity field as well as set up internal
       * data structures.
       */
      void
      reinit(
        std::shared_ptr<BoundaryDescriptor<dim, Number>> boundary_descriptor,
        std::shared_ptr<VelocityField>                   velocity_field,
        const AdvectionOperationParamters                additional_data)
      {
        this->factor_skew         = additional_data.factor_skew;
        this->boundary_descriptor = boundary_descriptor;
        this->velocity_field      = velocity_field;

        AssertDimension(
          (data.get_matrix_free_x().get_shape_info(0, 0).data[0].element_type ==
           dealii::internal::MatrixFreeFunctions::ElementType::
             tensor_symmetric_collocation),
          (data.get_matrix_free_v().get_shape_info(0, 0).data[0].element_type ==
           dealii::internal::MatrixFreeFunctions::ElementType::
             tensor_symmetric_collocation));

        const bool do_collocation =
          data.get_matrix_free_x().get_shape_info(0, 0).data[0].element_type ==
          dealii::internal::MatrixFreeFunctions::ElementType::
            tensor_symmetric_collocation;

        this->do_collocation = do_collocation;

        // clang-format off
        phi_cell.reset(new FEEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>(data, 0, 0, 0, 0));
        phi_cell_inv.reset(new FEEvaluationInverse<dim_x, dim_v, degree, n_points, Number, VNumber>(data, 0, 0, 0, 0));
        phi_cell_inv_co.reset(new FEEvaluationInverse<dim_x, dim_v, degree, degree + 1, Number, VNumber>(data, 0, 0, do_collocation ? 0 : 1, do_collocation ? 0 : 1));
        phi_face_m.reset(new FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>(data, true, 0, 0, 0, 0));
        phi_face_p.reset(new FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>(data, false, 0, 0, 0, 0));
        // clang-format on
      }

      /**
       * Apply operator. Depending on configuration ECL or FCL.
       */
      template <AdvectionOperationEvaluationLevel eval_level =
                  AdvectionOperationEvaluationLevel::all>
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
              &This::local_apply_advect_and_inverse_mass_matrix<eval_level>,
              this,
              dst,
              src,
              eval_level == AdvectionOperationEvaluationLevel::all ?
                MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
                  DataAccessOnFaces::values :
                MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
                  DataAccessOnFaces::none,
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
      template <AdvectionOperationEvaluationLevel eval_level =
                  AdvectionOperationEvaluationLevel::all>
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
                                                       VNumber,
                                                       Number>
          eval(*phi.get_shape_values(),
               dealii::AlignedVector<Number>(),
               dealii::AlignedVector<Number>());

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim - 1,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber,
                                                       Number>
          eval_face(*phi.get_shape_values(),
                    dealii::AlignedVector<Number>(),
                    dealii::AlignedVector<Number>());

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       n_points,
                                                       n_points,
                                                       VNumber,
                                                       Number>
          eval_(dealii::AlignedVector<Number>(),
                *phi.get_shape_gradients(),
                dealii::AlignedVector<Number>());

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber,
                                                       Number>
          eval_inv(dealii::AlignedVector<Number>(),
                   dealii::AlignedVector<Number>(),
                   data.get_matrix_free_x()
                     .get_shape_info()
                     .data[0]
                     .inverse_shape_values_eo);

        // clang-format off

        // 1) advection: cell contribution
        {
          this->velocity_field->reinit(cell);

          // load from global structure
          phi.reinit(cell);
          phi.read_dof_values(src);

          if(do_collocation == false)
            {
              if(degree + 1 == n_points)
                {
                  if (dim >= 1) eval.template values<0, true, false>(data_ptr, data_ptr);
                  if (dim >= 2) eval.template values<1, true, false>(data_ptr, data_ptr);
                  if (dim >= 3) eval.template values<2, true, false>(data_ptr, data_ptr);
                  if (dim >= 4) eval.template values<3, true, false>(data_ptr, data_ptr);
                  if (dim >= 5) eval.template values<4, true, false>(data_ptr, data_ptr);
                  if (dim >= 6) eval.template values<5, true, false>(data_ptr, data_ptr);
                } 
              else
                {
                  dealii::internal::FEEvaluationImplBasisChange<tensorproduct,
                                              dealii::internal::EvaluatorQuantity::value,
                                              dim,
                                              degree + 1,
                                              n_points>::
                    do_forward(1, data.get_matrix_free_x().get_shape_info().data.front().shape_values_eo,
                              data_ptr, 
                              data_ptr);   
                }
            }

          // copy quadrature values into buffer
          VNumber *buffer = phi_cell_inv->get_data_ptr();
          
          if(eval_level != AdvectionOperationEvaluationLevel::cell)
          for (auto i = 0u; i < dealii::Utilities::pow<unsigned int>(n_points, dim); i++)
            buffer[i] = data_ptr[i];

          // x-space
          {
            dealii::AlignedVector<VNumber> scratch_data_array;
            scratch_data_array.resize_fast(dealii::Utilities::pow(n_points, dim) * dim_x);
            VNumber *tempp = scratch_data_array.begin();
            
            if(factor_skew != 0.0)
              {
                if (dim_x >= 1) eval_.template gradients<0, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 0);
                if (dim_x >= 2) eval_.template gradients<1, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 1);
                if (dim_x >= 3) eval_.template gradients<2, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 2);
              }
            
            for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
              for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                {
                  VNumber    grad_in[dim_x];
                  const auto vel = velocity_field->evaluate_x(q, qx, qv);
                  
                  if(factor_skew != 0.0)
                    phi.template submit_value<false>(data_ptr, - factor_skew * (phi.get_gradient_x(tempp, q, qx, qv) * vel), q, qx, qv);
                  
                  if (factor_skew != 1.0)
                    {
                      for (int d = 0; d < dim_x; d++)
                        grad_in[d] = (1.0-factor_skew) * buffer[q] * vel[d];
                      phi.submit_gradient_x(tempp, grad_in, q, qx, qv);
                    }
                }
            
            if(factor_skew != 1.0)
              {
                if (dim_x >= 1 && (factor_skew != 0.0))
                  eval_.template gradients<0, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 0, data_ptr);
                else if (dim_x >= 1) 
                  eval_.template gradients<0, false, false>(tempp + dealii::Utilities::pow(n_points, dim) * 0, data_ptr);
                  
                if (dim_x >= 2) eval_.template gradients<1, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 1, data_ptr);
                if (dim_x >= 3) eval_.template gradients<2, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 2, data_ptr);
            }
          }
            
          // v-space
          {
            dealii::AlignedVector<VNumber> scratch_data_array;
            scratch_data_array.resize_fast(dealii::Utilities::pow(n_points, dim) * dim_v);
            VNumber *tempp = scratch_data_array.begin();
            
            if(factor_skew != 0.0)
              {
                if (dim_v >= 1) eval_.template gradients<0 + dim_x, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 0);
                if (dim_v >= 2) eval_.template gradients<1 + dim_x, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 1);
                if (dim_v >= 3) eval_.template gradients<2 + dim_x, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 2);
              }
            
            for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
              for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                {
                  VNumber    grad_in[dim_v];
                  const auto vel = velocity_field->evaluate_v(q, qx, qv);
                  
                  if(factor_skew != 0.0)
                    phi.template submit_value<true>(data_ptr, - factor_skew * (phi.get_gradient_v(tempp, q, qx, qv) * vel), q, qx, qv);
                  
                  if (factor_skew != 1.0)
                    {
                      for (int d = 0; d < dim_v; d++)
                        grad_in[d] = (1.0-factor_skew) * buffer[q] * vel[d];
                      phi.submit_gradient_v(tempp, grad_in, q, qx, qv);
                    }
                }

            if(factor_skew != 1.0)
              {
                if (dim_v >= 1) eval_.template gradients<0 + dim_x, false, true>(tempp + dealii::Utilities::pow(n_points, dim) * 0, data_ptr);
                if (dim_v >= 2) eval_.template gradients<1 + dim_x, false, true>(tempp + dealii::Utilities::pow(n_points, dim) * 1, data_ptr);
                if (dim_v >= 3) eval_.template gradients<2 + dim_x, false, true>(tempp + dealii::Utilities::pow(n_points, dim) * 2, data_ptr);
              }
          }
        }

        // 2) advection: faces
        if(eval_level != AdvectionOperationEvaluationLevel::cell)
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
              
              if(eval_level == AdvectionOperationEvaluationLevel::all)
                phi_p.read_dof_values(src);
            }

            if(do_collocation == false)
              {
                hyperdeal::internal::FEFaceNormalEvaluationImpl<dim_x, dim_v, n_points - 1, Number>::template interpolate_quadrature<true, false>(1, dealii::EvaluationFlags::values, data.get_matrix_free_x().get_shape_info(), /*out=*/data_ptr_inv, /*in=*/data_ptr1, face);
    
                if(degree + 1 == n_points)
                  {
                    if (dim >= 2) eval_face.template values<0, true, false>(data_ptr2, data_ptr2);
                    if (dim >= 3) eval_face.template values<1, true, false>(data_ptr2, data_ptr2);
                    if (dim >= 4) eval_face.template values<2, true, false>(data_ptr2, data_ptr2);
                    if (dim >= 5) eval_face.template values<3, true, false>(data_ptr2, data_ptr2);
                    if (dim >= 6) eval_face.template values<4, true, false>(data_ptr2, data_ptr2);
                  }
                  else
                  {
                    dealii::internal::FEEvaluationImplBasisChange<tensorproduct, 
                                                dealii::internal::EvaluatorQuantity::value, 
                                                dim - 1,
                                                degree + 1,
                                                n_points>::
                      do_forward(1 ,data.get_matrix_free_x().get_shape_info().data.front().shape_values_eo,
                                 data_ptr2, 
                                 data_ptr2);   
                  }
              }
            else
              {
                phi_m.read_dof_values_from_buffer(this->phi_cell_inv->get_data_ptr());
              }

            if(bid == dealii::numbers::internal_face_boundary_id)
              {
                if (face < dim_x * 2)
                  {
                    for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
                      for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x - 1); ++qx, ++q)
                        {
                          const VectorizedArrayType u_minus                     = data_ptr1[q];
                          const VectorizedArrayType u_plus                      = data_ptr2[q];
                          const VectorizedArrayType normal_times_speed          = velocity_field->evaluate_face_x(q, qx, qv) * phi_m.get_normal_vector_x(qx);
                          const VectorizedArrayType flux_times_normal_of_minus  = 0.5 * ((u_minus + u_plus) * normal_times_speed + std::abs(normal_times_speed) * (u_minus - u_plus)) * alpha;

                          phi_m.template submit_value<ID::SpaceType::X>(data_ptr1, flux_times_normal_of_minus - factor_skew*u_minus*normal_times_speed, q, qx, qv);
                        }
                  }
                else
                  {
                    for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v - 1); ++qv)
                      for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                        {
                          const VectorizedArrayType u_minus                     = data_ptr1[q];
                          const VectorizedArrayType u_plus                      = data_ptr2[q];
                          const VectorizedArrayType normal_times_speed          = velocity_field->evaluate_face_v(q, qx, qv) * phi_m.get_normal_vector_v(qv);
                          const VectorizedArrayType flux_times_normal_of_minus  = 0.5 * ((u_minus + u_plus) * normal_times_speed + std::abs(normal_times_speed) * (u_minus - u_plus)) * alpha;

                          phi_m.template submit_value<ID::SpaceType::V>(data_ptr1, flux_times_normal_of_minus - factor_skew*u_minus*normal_times_speed, q, qx, qv);
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
                              (-u_minus + 2.0 * hyperdeal::MatrixFreeTools::evaluate_scalar_function(phi_m.template get_quadrature_point<ID::SpaceType::X>(qx, qv), *boundary_pair.second, phi_m.n_vectorization_lanes_filled()));
                          
                          const VectorizedArrayType normal_times_speed          = velocity_field->evaluate_face_x(q, qx, qv) * phi_m.get_normal_vector_x(qx);
                          const VectorizedArrayType flux_times_normal_of_minus  = 0.5 * ((u_minus + u_plus) * normal_times_speed + std::abs(normal_times_speed) * (u_minus - u_plus)) * alpha;

                          phi_m.template submit_value<ID::SpaceType::X>(data_ptr1, flux_times_normal_of_minus - factor_skew*u_minus*normal_times_speed, q, qx, qv);
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
                              (-u_minus + 2.0 * hyperdeal::MatrixFreeTools::evaluate_scalar_function(phi_m.template get_quadrature_point<ID::SpaceType::V>(qx, qv), *boundary_pair.second, phi_m.n_vectorization_lanes_filled()));
                          
                          const VectorizedArrayType normal_times_speed          = velocity_field->evaluate_face_v(q, qx, qv) * phi_m.get_normal_vector_v(qv);
                          const VectorizedArrayType flux_times_normal_of_minus  = 0.5 * ((u_minus + u_plus) * normal_times_speed + std::abs(normal_times_speed) * (u_minus - u_plus)) * alpha;

                          phi_m.template submit_value<ID::SpaceType::V>(data_ptr1, flux_times_normal_of_minus - factor_skew*u_minus*normal_times_speed, q, qx, qv);
                        }
                  }
              }

            if(do_collocation == false)
              hyperdeal::internal::FEFaceNormalEvaluationImpl<dim_x, dim_v, n_points - 1, Number>::template interpolate_quadrature<false, true>(1, dealii::EvaluationFlags::values, data.get_matrix_free_x().get_shape_info(), /*out=*/data_ptr1, /*in=*/data_ptr, face);
            else
              phi_m.distribute_to_buffer(this->phi_cell->get_data_ptr());
          }

        // 3) inverse mass matrix
        {
          phi_inv.reinit(cell);

          for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
            for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
              phi_inv.submit_inv(data_ptr, q, qx, qv);

          if(do_collocation == false)
            {
              if(degree + 1 == n_points)
                {
                  if (dim >= 6) eval_inv.template hessians<5, false, false>(data_ptr, data_ptr);
                  if (dim >= 5) eval_inv.template hessians<4, false, false>(data_ptr, data_ptr);
                  if (dim >= 4) eval_inv.template hessians<3, false, false>(data_ptr, data_ptr);
                  if (dim >= 3) eval_inv.template hessians<2, false, false>(data_ptr, data_ptr);
                  if (dim >= 2) eval_inv.template hessians<1, false, false>(data_ptr, data_ptr);
                  if (dim >= 1) eval_inv.template hessians<0, false, false>(data_ptr, data_ptr);
                }
              else
                {
                  dealii::internal::FEEvaluationImplBasisChange<tensorproduct,
                                              dealii::internal::EvaluatorQuantity::hessian,
                                              dim,
                                              degree + 1,
                                              n_points>::
                    do_backward(1, data.get_matrix_free_x().get_shape_info().data.front().inverse_shape_values_eo,  
                                false,
                                data_ptr, 
                                data_ptr);   
                }
            }

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

        auto &phi_inv = *this->phi_cell_inv_co;

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
                                                       VectorizedArrayType,
                                                       Number>
          eval_inv(dealii::AlignedVector<Number>(),
                   dealii::AlignedVector<Number>(),
                   *phi_inv.get_inverse_shape());

        // clang-format off
        
        if(do_collocation == false)
          {
            if (dim >= 1) eval_inv.template hessians<0, true, false>(data_ptr, data_ptr);
            if (dim >= 2) eval_inv.template hessians<1, true, false>(data_ptr, data_ptr);
            if (dim >= 3) eval_inv.template hessians<2, true, false>(data_ptr, data_ptr);
            if (dim >= 4) eval_inv.template hessians<3, true, false>(data_ptr, data_ptr);
            if (dim >= 5) eval_inv.template hessians<4, true, false>(data_ptr, data_ptr);
            if (dim >= 6) eval_inv.template hessians<5, true, false>(data_ptr, data_ptr);
          }

        for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(degree + 1, dim_v); ++qv)
          for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(degree + 1, dim_x); ++qx, ++q)
            phi_inv.submit_inv(data_ptr, q, qx, qv);

        if(do_collocation == false)
          {
            if (dim >= 6) eval_inv.template hessians<5, false, false>(data_ptr, data_ptr);
            if (dim >= 5) eval_inv.template hessians<4, false, false>(data_ptr, data_ptr);
            if (dim >= 4) eval_inv.template hessians<3, false, false>(data_ptr, data_ptr);
            if (dim >= 3) eval_inv.template hessians<2, false, false>(data_ptr, data_ptr);
            if (dim >= 2) eval_inv.template hessians<1, false, false>(data_ptr, data_ptr);
            if (dim >= 1) eval_inv.template hessians<0, false, false>(data_ptr, data_ptr);
          }

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
                                                       VNumber,
                                                       Number>
          eval(*phi.get_shape_values(),
               dealii::AlignedVector<Number>(),
               dealii::AlignedVector<Number>());

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       n_points,
                                                       n_points,
                                                       VNumber,
                                                       Number>
          eval_(dealii::AlignedVector<Number>(),
                *phi.get_shape_gradients(),
                dealii::AlignedVector<Number>());

        // clang-format off

        this->velocity_field->reinit(cell);

        // load from global structure
        phi.reinit(cell);
        phi.read_dof_values(src);

        if(do_collocation == false)
          {
            if(degree + 1 == n_points)
              {
                if (dim >= 1) eval.template values<0, true, false>(data_ptr, data_ptr);
                if (dim >= 2) eval.template values<1, true, false>(data_ptr, data_ptr);
                if (dim >= 3) eval.template values<2, true, false>(data_ptr, data_ptr);
                if (dim >= 4) eval.template values<3, true, false>(data_ptr, data_ptr);
                if (dim >= 5) eval.template values<4, true, false>(data_ptr, data_ptr);
                if (dim >= 6) eval.template values<5, true, false>(data_ptr, data_ptr);
              }
              else
              {
                dealii::internal::FEEvaluationImplBasisChange<tensorproduct,
                                            dealii::internal::EvaluatorQuantity::value,
                                            dim,
                                            degree + 1,
                                            n_points>::
                  do_forward(1, data.get_matrix_free_x().get_shape_info().data.front().shape_values_eo,
                             data_ptr, 
                             data_ptr);   
              }
          }

        // copy quadrature values into buffer
        VNumber *buffer = phi_cell_inv->get_data_ptr();
        for (auto i = 0u; i < dealii::Utilities::pow<unsigned int>(n_points, dim); i++)
          buffer[i] = data_ptr[i];

        // x-space
        {
          dealii::AlignedVector<VNumber> scratch_data_array;
          scratch_data_array.resize_fast(dealii::Utilities::pow(n_points, dim) * dim_x);
          VNumber *tempp = scratch_data_array.begin();
            
            if(factor_skew != 0.0)
              {
                if (dim_x >= 1) eval_.template gradients<0, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 0);
                if (dim_x >= 2) eval_.template gradients<1, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 1);
                if (dim_x >= 3) eval_.template gradients<2, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 2);
              }
          
          for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
            for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
              {
                  VNumber    grad_in[dim_x];
                  const auto vel = velocity_field->evaluate_x(q, qx, qv);
                  
                  if(factor_skew != 0.0)
                    phi.template submit_value<false>(data_ptr, - factor_skew * (phi.get_gradient_x(tempp, q, qx, qv) * vel), q, qx, qv);
                  
                  if (factor_skew != 1.0)
                    {
                      for (int d = 0; d < dim_x; d++)
                        grad_in[d] = (1.0-factor_skew) * buffer[q] * vel[d];
                      phi.submit_gradient_x(tempp, grad_in, q, qx, qv);
                    }
              }
          
          if(factor_skew != 1.0)
            {
              if (dim_x >= 1 && (factor_skew != 0.0))
                eval_.template gradients<0, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 0, data_ptr);
              else if (dim_x >= 1) 
                eval_.template gradients<0, false, false>(tempp + dealii::Utilities::pow(n_points, dim) * 0, data_ptr);
              if (dim_x >= 2) eval_.template gradients<1, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 1, data_ptr);
              if (dim_x >= 3) eval_.template gradients<2, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 2, data_ptr);
            }
        }
        // v-space
        {
          dealii::AlignedVector<VNumber> scratch_data_array;
          scratch_data_array.resize_fast(dealii::Utilities::pow(n_points, dim) * dim_v);
          VNumber *tempp = scratch_data_array.begin();
            
          if(factor_skew != 0.0)
            {
              if (dim_v >= 1) eval_.template gradients<0 + dim_x, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 0);
              if (dim_v >= 2) eval_.template gradients<1 + dim_x, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 1);
              if (dim_v >= 3) eval_.template gradients<2 + dim_x, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 2);
            }
          
          for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
            for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
              {
                  VNumber    grad_in[dim_v];
                  const auto vel = velocity_field->evaluate_v(q, qx, qv);
                  
                  if(factor_skew != 0.0)
                    phi.template submit_value<true>(data_ptr, - factor_skew * (phi.get_gradient_v(tempp, q, qx, qv) * vel), q, qx, qv);
                  
                  if (factor_skew != 1.0)
                    {
                      for (int d = 0; d < dim_v; d++)
                        grad_in[d] = (1.0-factor_skew) * buffer[q] * vel[d];
                      phi.submit_gradient_v(tempp, grad_in, q, qx, qv);
                    }
              }

          if(factor_skew != 1.0)
            {
              if (dim_v >= 1) eval_.template gradients<0 + dim_x, false, true>(tempp + dealii::Utilities::pow(n_points, dim) * 0, data_ptr);
              if (dim_v >= 2) eval_.template gradients<1 + dim_x, false, true>(tempp + dealii::Utilities::pow(n_points, dim) * 1, data_ptr);
              if (dim_v >= 3) eval_.template gradients<2 + dim_x, false, true>(tempp + dealii::Utilities::pow(n_points, dim) * 2, data_ptr);
            }
        }

        if(do_collocation == false)
          {
            if(degree + 1 == n_points)
              {
                if(dim >= 6) eval.template values<5, false, false>(data_ptr, data_ptr);
                if(dim >= 5) eval.template values<4, false, false>(data_ptr, data_ptr);
                if(dim >= 4) eval.template values<3, false, false>(data_ptr, data_ptr);
                if(dim >= 3) eval.template values<2, false, false>(data_ptr, data_ptr);
                if(dim >= 2) eval.template values<1, false, false>(data_ptr, data_ptr);
                if(dim >= 1) eval.template values<0, false, false>(data_ptr, data_ptr);
              }
              else
              {
                dealii::internal::FEEvaluationImplBasisChange<tensorproduct,
                                            dealii::internal::EvaluatorQuantity::value,
                                            dim,
                                            degree + 1,
                                            n_points>::
                  do_backward(1, data.get_matrix_free_x().get_shape_info().data.front().shape_values_eo, 
                             false,
                             data_ptr, 
                             data_ptr);   
              }
          }

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
                                                       VNumber,
                                                       Number>
          eval1(*phi_m.get_shape_values(),
                dealii::AlignedVector<Number>(),
                dealii::AlignedVector<Number>());
        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim - 1,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber,
                                                       Number>
          eval2(*phi_p.get_shape_values(),
                dealii::AlignedVector<Number>(),
                dealii::AlignedVector<Number>());

        this->velocity_field->reinit_face(face);

        // get data and scratch
        VNumber *data_ptr1 = phi_m.get_data_ptr();
        VNumber *data_ptr2 = phi_p.get_data_ptr();

        // load from global structure
        phi_m.reinit(face);
        phi_p.reinit(face);

        // clang-format off

        phi_m.read_dof_values(src);
        
        if(do_collocation == false)
          {
            if(degree + 1 == n_points)
              {
                if (dim >= 2) eval1.template values<0, true, false>(data_ptr1, data_ptr1);
                if (dim >= 3) eval1.template values<1, true, false>(data_ptr1, data_ptr1);
                if (dim >= 4) eval1.template values<2, true, false>(data_ptr1, data_ptr1);
                if (dim >= 5) eval1.template values<3, true, false>(data_ptr1, data_ptr1);
                if (dim >= 6) eval1.template values<4, true, false>(data_ptr1, data_ptr1);
              }
              else
              {
                dealii::internal::FEEvaluationImplBasisChange<tensorproduct,
                                            dealii::internal::EvaluatorQuantity::value, 
                                            dim - 1,
                                            degree + 1,
                                            n_points>::
                  do_forward(1, data.get_matrix_free_x().get_shape_info().data.front().shape_values_eo,
                             data_ptr1, 
                             data_ptr1);   
              }
          }

        phi_p.read_dof_values(src);
        
        if(do_collocation == false)
          {
            if(degree + 1 == n_points)
              {
                if (dim >= 2) eval2.template values<0, true, false>(data_ptr2, data_ptr2);
                if (dim >= 3) eval2.template values<1, true, false>(data_ptr2, data_ptr2);
                if (dim >= 4) eval2.template values<2, true, false>(data_ptr2, data_ptr2);
                if (dim >= 5) eval2.template values<3, true, false>(data_ptr2, data_ptr2);
                if (dim >= 6) eval2.template values<4, true, false>(data_ptr2, data_ptr2);
              }
              else
              {
                dealii::internal::FEEvaluationImplBasisChange<tensorproduct,
                                            dealii::internal::EvaluatorQuantity::value,
                                            dim - 1,
                                            degree + 1,
                                            n_points>::
                  do_forward(1, data.get_matrix_free_x().get_shape_info().data.front().shape_values_eo,
                             data_ptr2, 
                             data_ptr2);   
              }
          }

        if (face.type == ID::SpaceType::X)
          {
            for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
              for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x - 1); ++qx, ++q)
                {
                  const VectorizedArrayType u_minus                     = data_ptr1[q];
                  const VectorizedArrayType u_plus                      = data_ptr2[q];
                  const VectorizedArrayType normal_times_speed          = velocity_field->evaluate_face_x(q, qx, qv) * phi_m.get_normal_vector_x(qx);
                  const VectorizedArrayType flux_times_normal_of_minus  = 0.5 * ((u_minus + u_plus) * normal_times_speed + std::abs(normal_times_speed) * (u_minus - u_plus)) * alpha;

                  phi_m.template submit_value<ID::SpaceType::X>(data_ptr1, +flux_times_normal_of_minus - factor_skew*u_minus*normal_times_speed, q, qx, qv);
                  phi_p.template submit_value<ID::SpaceType::X>(data_ptr2, -flux_times_normal_of_minus + factor_skew*u_plus*normal_times_speed, q, qx, qv);
                }
          }
        else
          {
            for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v - 1); ++qv)
              for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                {
                  const VectorizedArrayType u_minus                     = data_ptr1[q]; 
                  const VectorizedArrayType u_plus                      = data_ptr2[q];
                  const VectorizedArrayType normal_times_speed          = velocity_field->evaluate_face_v(q, qx, qv) * phi_m.get_normal_vector_v(qv);
                  const VectorizedArrayType flux_times_normal_of_minus  = 0.5 * ((u_minus + u_plus) * normal_times_speed + std::abs(normal_times_speed) * (u_minus - u_plus)) * alpha;

                  phi_m.template submit_value<ID::SpaceType::V>(data_ptr1, +flux_times_normal_of_minus - factor_skew*u_minus*normal_times_speed, q, qx, qv);
                  phi_p.template submit_value<ID::SpaceType::V>(data_ptr2, -flux_times_normal_of_minus + factor_skew*u_plus*normal_times_speed, q, qx, qv);
                }
          }

        if(do_collocation == false)
          {
            if(degree + 1 == n_points)
              {
                if (dim >= 6) eval1.template values<4, false, false>(data_ptr1, data_ptr1);
                if (dim >= 5) eval1.template values<3, false, false>(data_ptr1, data_ptr1);
                if (dim >= 4) eval1.template values<2, false, false>(data_ptr1, data_ptr1);
                if (dim >= 3) eval1.template values<1, false, false>(data_ptr1, data_ptr1);
                if (dim >= 2) eval1.template values<0, false, false>(data_ptr1, data_ptr1);
              }
              else
              {
                dealii::internal::FEEvaluationImplBasisChange<tensorproduct,
                                            dealii::internal::EvaluatorQuantity::value,
                                            dim - 1,
                                            degree + 1,
                                            n_points>::
                  do_backward(1, data.get_matrix_free_x().get_shape_info().data.front().shape_values_eo, false,
                             data_ptr1, 
                             data_ptr1);   
              }
          }

        // write into global structure back
        phi_m.distribute_local_to_global(dst);

        if(do_collocation == false)
          {
            if(degree + 1 == n_points)
              {
                if (dim >= 6) eval2.template values<4, false, false>(data_ptr2, data_ptr2);
                if (dim >= 5) eval2.template values<3, false, false>(data_ptr2, data_ptr2);
                if (dim >= 4) eval2.template values<2, false, false>(data_ptr2, data_ptr2);
                if (dim >= 3) eval2.template values<1, false, false>(data_ptr2, data_ptr2);
                if (dim >= 2) eval2.template values<0, false, false>(data_ptr2, data_ptr2);
              }
              else
              {
                dealii::internal::FEEvaluationImplBasisChange<tensorproduct,
                                            dealii::internal::EvaluatorQuantity::value,
                                            dim - 1,
                                            degree + 1,
                                            n_points>::
                  do_backward(1, data.get_matrix_free_x().get_shape_info().data.front().shape_values_eo, false,
                             data_ptr2, 
                             data_ptr2);   
              }
          }

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
                                                       VNumber,
                                                       Number>
          eval1(*phi_m.get_shape_values(),
                dealii::AlignedVector<Number>(),
                dealii::AlignedVector<Number>());

        this->velocity_field->reinit_face(face);

        // get data and scratch
        VNumber *data_ptr1 = phi_m.get_data_ptr();

        // load from global structure
        phi_m.reinit(face);

        // clang-format off

        phi_m.read_dof_values(src);
        
        if(do_collocation == false)
          {
            if(degree + 1 == n_points)
              {
                if (dim >= 2) eval1.template values<0, true, false>(data_ptr1, data_ptr1);
                if (dim >= 3) eval1.template values<1, true, false>(data_ptr1, data_ptr1);
                if (dim >= 4) eval1.template values<2, true, false>(data_ptr1, data_ptr1);
                if (dim >= 5) eval1.template values<3, true, false>(data_ptr1, data_ptr1);
                if (dim >= 6) eval1.template values<4, true, false>(data_ptr1, data_ptr1);
              }
              else
              {
                dealii::internal::FEEvaluationImplBasisChange<tensorproduct,
                                            dealii::internal::EvaluatorQuantity::value,
                                            dim - 1,
                                            degree + 1,
                                            n_points>::
                  do_forward(1, data.get_matrix_free_x().get_shape_info().data.front().shape_values_eo,
                             data_ptr1, 
                             data_ptr1);   
              }
          }

        if (face.type == ID::SpaceType::X)
          {
            for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
              for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x - 1); ++qx, ++q)
                {
                  const VectorizedArrayType u_minus = data_ptr1[q];
                  const VectorizedArrayType u_plus = (boundary_pair.first == BoundaryType::DirichletHomogenous) ? 
                      (-u_minus) : 
                      (-u_minus + 2.0 * hyperdeal::MatrixFreeTools::evaluate_scalar_function(phi_m.template get_quadrature_point<ID::SpaceType::X>(qx, qv), *boundary_pair.second, phi_m.n_vectorization_lanes_filled()));
                  
                  const VectorizedArrayType normal_times_speed          = velocity_field->evaluate_face_x(q, qx, qv) * phi_m.get_normal_vector_x(qx);
                  const VectorizedArrayType flux_times_normal_of_minus  = 0.5 * ((u_minus + u_plus) * normal_times_speed + std::abs(normal_times_speed) * (u_minus - u_plus)) * alpha;

                  phi_m.template submit_value<ID::SpaceType::X>(data_ptr1, flux_times_normal_of_minus - factor_skew*u_minus*normal_times_speed, q, qx, qv);
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
                      (-u_minus + 2.0 * hyperdeal::MatrixFreeTools::evaluate_scalar_function(phi_m.template get_quadrature_point<ID::SpaceType::V>(qx, qv), *boundary_pair.second, phi_m.n_vectorization_lanes_filled()));
                  
                  const VectorizedArrayType normal_times_speed          = velocity_field->evaluate_face_v(q, qx, qv) * phi_m.get_normal_vector_v(qv);
                  const VectorizedArrayType flux_times_normal_of_minus  = 0.5 * ((u_minus + u_plus) * normal_times_speed + std::abs(normal_times_speed) * (u_minus - u_plus)) * alpha;

                  phi_m.template submit_value<ID::SpaceType::V>(data_ptr1, flux_times_normal_of_minus - factor_skew*u_minus*normal_times_speed, q, qx, qv);
                }
          }

        if(do_collocation == false)
          {
            if(degree + 1 == n_points)
              {
                if (dim >= 6) eval1.template values<4, false, false>(data_ptr1, data_ptr1);
                if (dim >= 5) eval1.template values<3, false, false>(data_ptr1, data_ptr1);
                if (dim >= 4) eval1.template values<2, false, false>(data_ptr1, data_ptr1);
                if (dim >= 3) eval1.template values<1, false, false>(data_ptr1, data_ptr1);
                if (dim >= 2) eval1.template values<0, false, false>(data_ptr1, data_ptr1);
              }
              else
              {
                dealii::internal::FEEvaluationImplBasisChange<tensorproduct,
                                            dealii::internal::EvaluatorQuantity::value,
                                            dim - 1,
                                            degree + 1,
                                            n_points>::
                  do_backward(1, data.get_matrix_free_x().get_shape_info().data.front().shape_values_eo, 
                             false,
                             data_ptr1, 
                             data_ptr1);   
              }
          }

        // write into global structure back
        phi_m.distribute_local_to_global(dst);
      }

      const MatrixFree<dim_x, dim_v, Number, VectorizedArrayType> &data;
      DynamicConvergenceTable &                                    table;

      // clang-format off
      std::shared_ptr<FEEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>> phi_cell;
      std::shared_ptr<FEEvaluationInverse<dim_x, dim_v, degree, n_points, Number, VNumber>> phi_cell_inv;
      std::shared_ptr<FEEvaluationInverse<dim_x, dim_v, degree, degree + 1, Number, VNumber>> phi_cell_inv_co;
      std::shared_ptr<FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>> phi_face_m;
      std::shared_ptr<FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>> phi_face_p;
      // clang-format on

      std::shared_ptr<BoundaryDescriptor<dim, Number>> boundary_descriptor;
      std::shared_ptr<VelocityField>                   velocity_field;

      bool do_collocation;

      const double alpha = 1.0;

      // skew factor: conservative (skew=0) and convective (skew=1)
      double factor_skew = 0.0;
    };
  } // namespace advection
} // namespace hyperdeal


#endif
