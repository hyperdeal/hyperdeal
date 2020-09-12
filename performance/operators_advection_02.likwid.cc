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

// Test performance of components of advection operator run as a stand alone.


#define NUMBER_TYPE double
#define MIN_DEGREE 3
#define MAX_DEGREE 5
#define MIN_DIM 2
#define MAX_DIM 6
#define MIN_SIMD_LENGTH 0
#define MAX_SIMD_LENGTH 0

#include <hyper.deal/grid/grid_generator.h>
#include <hyper.deal/operators/advection/cfl.h>
#include <hyper.deal/operators/advection/velocity_field_view.h>

#include <deal.II/matrix_free/evaluation_kernels.h>

#include <hyper.deal/base/dynamic_convergence_table.h>
#include <hyper.deal/matrix_free/fe_evaluation_cell.h>
#include <hyper.deal/matrix_free/fe_evaluation_cell_inverse.h>
#include <hyper.deal/matrix_free/fe_evaluation_face.h>
#include <hyper.deal/matrix_free/matrix_free.h>
#include <hyper.deal/matrix_free/tools.h>
#include <hyper.deal/operators/advection/advection_operation_parameters.h>
#include <hyper.deal/operators/advection/boundary_descriptor.h>

#include "../tests/tests_mf.h"
#include "util/driver.h"


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
    class AdvectionOperation1
    {
    public:
      using This    = AdvectionOperation1<dim_x,
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
      AdvectionOperation1(
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
        
        (void) velocity_field;

        AssertDimension(
          (data.get_matrix_free_x().get_shape_info(0, 0).data[0].element_type ==
           dealii::internal::MatrixFreeFunctions::ElementType::
             tensor_symmetric_collocation),
          (data.get_matrix_free_v().get_shape_info(0, 0).data[0].element_type ==
           dealii::internal::MatrixFreeFunctions::ElementType::
             tensor_symmetric_collocation))

          const bool do_collocation =
            data.get_matrix_free_x()
              .get_shape_info(0, 0)
              .data[0]
              .element_type == dealii::internal::MatrixFreeFunctions::
                                 ElementType::tensor_symmetric_collocation;

        this->do_collocation = do_collocation;

        // clang-format off
        phi_cell.reset(new FEEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>(data, 0, 0, 0, 0));
        phi_cell_inv.reset(new FEEvaluationInverse<dim_x, dim_v, degree, n_points, Number, VNumber>(data, 0, 0, 0, 0));
        phi_cell_inv_co.reset(new FEEvaluationInverse<dim_x, dim_v, degree, degree + 1, Number, VNumber>(data, 0, 0, do_collocation ? 0 : 1, do_collocation ? 0 : 1));
        phi_face_m.reset(new FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>(data, true, 0, 0, 0, 0));
        phi_face_p.reset(new FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>(data, false, 0, 0, 0, 0));
        // clang-format on
        
        const unsigned int n_cells   = data.get_matrix_free_x().n_cell_batches() * data.get_matrix_free_v().n_cell_batches();
        const unsigned int n_q_cells = n_cells * dealii::Utilities::pow(n_points, dim);
        const unsigned int n_q_faces = 2*n_cells * dim * dealii::Utilities::pow(n_points, dim - 1);

        JxW_values.resize(n_q_cells, 0);
        JxW_inv_values.resize(n_q_cells, 0);
        inverse_jacobian_values.resize(n_q_cells, Tensor<2,dim,typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX>());

        JxW_face_values.resize(n_q_faces, 0);
        normal_values.resize(n_q_faces, Tensor<1,dim,typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX>() );
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
            Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
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
                                                       VNumber>
          eval(*phi.get_shape_values(),
               dealii::AlignedVector<VNumber>(),
               dealii::AlignedVector<VNumber>());

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim - 1,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber>
          eval_face(*phi.get_shape_values(),
                    dealii::AlignedVector<VNumber>(),
                    dealii::AlignedVector<VNumber>());

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       n_points,
                                                       n_points,
                                                       VNumber>
          eval_(dealii::AlignedVector<VNumber>(),
                *phi.get_shape_gradients(),
                dealii::AlignedVector<VNumber>());

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber>
          eval_inv(dealii::AlignedVector<VNumber>(),
                   dealii::AlignedVector<VNumber>(),
                   data.get_matrix_free_x()
                     .get_shape_info()
                     .data[0]
                     .inverse_shape_values_eo);

        dealii::Tensor<1, dim, VectorizedArrayType> vel; // constant velocity
        
        {
            const unsigned int offset = cell.macro * dealii::Utilities::pow(n_points, dim_x + dim_v);
            JxW              = &JxW_values[offset];
            JxW_inv          = &JxW_inv_values[offset];
            inverse_jacobian = &inverse_jacobian_values[offset];
        }
        // clang-format off
        // 1) advection: cell contribution
        {

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
                  Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
                }
            }
          else
            {
              Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
            }

          // copy quadrature values into buffer
          VNumber *buffer = phi_cell_inv->get_data_ptr();
          
          if(eval_level != AdvectionOperationEvaluationLevel::cell)
          for (auto i = 0u; i < dealii::Utilities::pow<unsigned int>(n_points, dim); i++)
            buffer[i] = data_ptr[i];

          // xv-space
          {
            dealii::AlignedVector<VNumber> scratch_data_array;
            scratch_data_array.resize_fast(dealii::Utilities::pow(n_points, dim) * dim);
            VNumber *tempp = scratch_data_array.begin();
            
            if(factor_skew != 0.0)
              {
                if (dim >= 1) eval_.template gradients<0, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 0);
                if (dim >= 2) eval_.template gradients<1, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 1);
                if (dim >= 3) eval_.template gradients<2, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 2);
                if (dim >= 4) eval_.template gradients<3, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 3);
                if (dim >= 5) eval_.template gradients<4, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 4);
                if (dim >= 6) eval_.template gradients<5, true, false>(buffer, tempp + dealii::Utilities::pow(n_points, dim) * 5);
              }
            
            for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
              for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                {
                  VNumber    grad_in[dim];
                  
                  if(factor_skew != 0.0)
                    submit_value_cell<false>(data_ptr, - factor_skew * (get_gradient_cell(tempp, q, qx, qv) * vel), q, qx, qv);
                  
                  if (factor_skew != 1.0)
                    {
                      for (int d = 0; d < dim; d++)
                        grad_in[d] = (1.0-factor_skew) * buffer[q] * vel[d];
                      submit_gradient_cell(tempp, grad_in, q, qx, qv);
                    }
                }
            
            if(factor_skew != 1.0)
              {
                if (dim >= 1 && (factor_skew != 0.0))
                  eval_.template gradients<0, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 0, data_ptr);
                else if (dim >= 1) 
                  eval_.template gradients<0, false, false>(tempp + dealii::Utilities::pow(n_points, dim) * 0, data_ptr);
                  
                if (dim >= 2) eval_.template gradients<1, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 1, data_ptr);
                if (dim >= 3) eval_.template gradients<2, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 2, data_ptr);
                if (dim >= 4) eval_.template gradients<2, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 3, data_ptr);
                if (dim >= 5) eval_.template gradients<2, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 4, data_ptr);
                if (dim >= 6) eval_.template gradients<2, false, true >(tempp + dealii::Utilities::pow(n_points, dim) * 5, data_ptr);
            }
          }
        }

        // 2) advection: faces
        if(eval_level != AdvectionOperationEvaluationLevel::cell)
        for (auto face = 0u; face < dim * 2; face++)
          {
            // load negative side from buffer
            phi_m.reinit(cell, face);
            
            {
              const unsigned int offset_face = (cell.macro  * 2 * dim + face) * dealii::Utilities::pow(n_points, dim_x + dim_v - 1);
              JxW_face = &JxW_face_values[offset_face];
              normal   = &normal_values[offset_face];
            }
            
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
                dealii::internal::FEFaceNormalEvaluationImpl<dim, n_points - 1, 1, VectorizedArrayType, true>::template interpolate_quadrature<true, false>(data.get_matrix_free_x().get_shape_info(), /*out=*/data_ptr_inv, /*in=*/data_ptr1, false, face);
    
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
                    Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
                  }
              }
            else
              {
                Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
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
                          const VectorizedArrayType normal_times_speed          = vel * get_normal_vector_face(q);
                          const VectorizedArrayType flux_times_normal_of_minus  = 0.5 * ((u_minus + u_plus) * normal_times_speed + std::abs(normal_times_speed) * (u_minus - u_plus)) * alpha;

                          submit_value_face<ID::SpaceType::X>(data_ptr1, flux_times_normal_of_minus - factor_skew*u_minus*normal_times_speed, q, qx, qv);
                        }
                  }
                else
                  {
                    for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v - 1); ++qv)
                      for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                        {
                          const VectorizedArrayType u_minus                     = data_ptr1[q];
                          const VectorizedArrayType u_plus                      = data_ptr2[q];
                          const VectorizedArrayType normal_times_speed          = vel * get_normal_vector_face(q);
                          const VectorizedArrayType flux_times_normal_of_minus  = 0.5 * ((u_minus + u_plus) * normal_times_speed + std::abs(normal_times_speed) * (u_minus - u_plus)) * alpha;

                          submit_value_face<ID::SpaceType::V>(data_ptr1, flux_times_normal_of_minus - factor_skew*u_minus*normal_times_speed, q, qx, qv);
                        }
                  }
              }
            else
              {
                Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
              }

            if(do_collocation == false)
              dealii::internal::FEFaceNormalEvaluationImpl<dim, n_points - 1, 1, VectorizedArrayType, true>::template interpolate_quadrature<false, true>(data.get_matrix_free_x().get_shape_info(), /*out=*/data_ptr1, /*in=*/data_ptr, false, face);
            else
            {
                Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
            }
          }

        // 3) inverse mass matrix
        {
          phi_inv.reinit(cell);

          for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
            for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
              submit_inv(data_ptr, q, qx, qv);

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
                  Assert(false, dealii::StandardExceptions::ExcNotImplemented ());  
                }
            }
          else
            {
              Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
            }

          // write into global structure back
          phi.set_dof_values(dst);
        }

        // clang-format on
      }
      
      template<bool do_add>
      void
      submit_value_cell(VectorizedArrayType *__restrict data_ptr_out,
                   const VectorizedArrayType __restrict values_in,
                   const unsigned int q,
                   const unsigned int qx,
                   const unsigned int qv) const
      {
        (void) qx;
        (void) qv;
          
        const auto jxw = JxW_inv[q];

        if (do_add)
          data_ptr_out[q] += values_in * jxw;
        else
          data_ptr_out[q] = values_in * jxw;
      }
      
      void
      submit_inv(VectorizedArrayType *__restrict data_ptr_out,
                 const unsigned int q,
                 const unsigned int qx,
                 const unsigned int qv) const
      {
        (void) qx;
        (void) qv;
          
        data_ptr_out[q] *= JxW_inv[q];
      }
      
      
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<1, dim, VectorizedArrayType>
      get_gradient_cell(const VectorizedArrayType *__restrict grad_in,
                     const unsigned int q,
                     const unsigned int qx,
                     const unsigned int qv) const
    {
        (void) qx;
        (void) qv;
          
        const auto jxw        = JxW[q];
        const auto & jacobian = inverse_jacobian[qx];

      dealii::Tensor<1, dim, VectorizedArrayType> result;

      for (auto d = 0u; d < dim; d++)
        {
          result[d] = jacobian[d][0] * grad_in[q];
          for (auto e = 1u; e < dim; ++e)
            result[d] +=
              (jacobian[d][e] * grad_in[q + e * dealii::Utilities::pow<unsigned int>(n_points, dim_x) * dealii::Utilities::pow<unsigned int>(n_points, dim_v)]);
        }

      return result;
    }
    
      
      inline DEAL_II_ALWAYS_INLINE //
        void
        submit_gradient_cell(VectorizedArrayType *__restrict data_ptr_out,
                          const VectorizedArrayType *__restrict grad_in,
                          const unsigned int q,
                          const unsigned int qx,
                          const unsigned int qv) const
      {
        (void) qx;
        (void) qv;
          
        const auto jxw        = JxW[q];
        const auto & jacobian = inverse_jacobian[qx];

  
        for (auto d = 0u; d < dim; d++)
          {
            auto new_val = jacobian[0][d] * grad_in[0];
            for (auto e = 1u; e < dim; ++e)
              new_val += (jacobian[e][d] * grad_in[e]);
            data_ptr_out[q + d * dealii::Utilities::pow<unsigned int>(n_points, dim_x) * dealii::Utilities::pow<unsigned int>(n_points, dim_v)] = new_val * jxw;
          }
      }
      
      
    template <TensorID::SpaceType stype>
    inline DEAL_II_ALWAYS_INLINE //
      void
      submit_value_face(VectorizedArrayType *__restrict data_ptr,
                   const VectorizedArrayType &value,
                   const unsigned int         q,
                   const unsigned int         qx,
                   const unsigned int         qv) const
    {
        (void) qx;
        (void) qv;
        
      data_ptr[q] = -value * JxW_face[q];
    }
      
      
    inline DEAL_II_ALWAYS_INLINE //
      dealii::Tensor<1, dim, VectorizedArrayType>
      get_normal_vector_face(const unsigned int q) const
    {
      return normal[q];
    }
    
      AlignedVector<typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX> JxW_values;
      AlignedVector<typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX> JxW_inv_values;
      AlignedVector<typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX> JxW_face_values;
      AlignedVector<Tensor<2,dim,typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX>> inverse_jacobian_values;
      AlignedVector<Tensor<1,dim,typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX>> normal_values;
    
      typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX* JxW;
      typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX* JxW_inv;
      typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX* JxW_face;
      Tensor<2,dim,typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX>* inverse_jacobian;
      Tensor<1,dim,typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX>* normal;


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

      bool do_collocation;

      const double alpha = 1.0;

      // skew factor: conservative (skew=0) and convective (skew=1)
      double factor_skew = 0.0;
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
    class AdvectionOperation2
    {
    public:
      using This    = AdvectionOperation2<dim_x,
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
      AdvectionOperation2(
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
        (void) velocity_field;
          
        this->factor_skew         = additional_data.factor_skew;
        this->boundary_descriptor = boundary_descriptor;

        AssertDimension(
          (data.get_matrix_free_x().get_shape_info(0, 0).data[0].element_type ==
           dealii::internal::MatrixFreeFunctions::ElementType::
             tensor_symmetric_collocation),
          (data.get_matrix_free_v().get_shape_info(0, 0).data[0].element_type ==
           dealii::internal::MatrixFreeFunctions::ElementType::
             tensor_symmetric_collocation))

          const bool do_collocation =
            data.get_matrix_free_x()
              .get_shape_info(0, 0)
              .data[0]
              .element_type == dealii::internal::MatrixFreeFunctions::
                                 ElementType::tensor_symmetric_collocation;

        this->do_collocation = do_collocation;

        // clang-format off
        phi_cell.reset(new FEEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>(data, 0, 0, 0, 0));
        phi_cell_inv.reset(new FEEvaluationInverse<dim_x, dim_v, degree, n_points, Number, VNumber>(data, 0, 0, 0, 0));
        phi_cell_inv_co.reset(new FEEvaluationInverse<dim_x, dim_v, degree, degree + 1, Number, VNumber>(data, 0, 0, do_collocation ? 0 : 1, do_collocation ? 0 : 1));
        phi_face_m.reset(new FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>(data, true, 0, 0, 0, 0));
        phi_face_p.reset(new FEFaceEvaluation<dim_x, dim_v, degree, n_points, Number, VNumber>(data, false, 0, 0, 0, 0));
        // clang-format on
        
        
        const unsigned int n_cells   = 1;
        const unsigned int n_q_cells = n_cells * dealii::Utilities::pow(n_points, dim);
        const unsigned int n_q_faces = 2*n_cells * dim * dealii::Utilities::pow(n_points, dim - 1);

        JxW_values.resize(n_q_cells, 0);
        JxW_inv_values.resize(n_q_cells, 0);
        inverse_jacobian_values.resize(n_q_cells, Tensor<2,dim,typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX>());

        JxW_face_values.resize(n_q_faces, 0);
        
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
            Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
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
                                                       VNumber>
          eval(*phi.get_shape_values(),
               dealii::AlignedVector<VNumber>(),
               dealii::AlignedVector<VNumber>());

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim - 1,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber>
          eval_face(*phi.get_shape_values(),
                    dealii::AlignedVector<VNumber>(),
                    dealii::AlignedVector<VNumber>());

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       n_points,
                                                       n_points,
                                                       VNumber>
          eval_(dealii::AlignedVector<VNumber>(),
                *phi.get_shape_gradients(),
                dealii::AlignedVector<VNumber>());

        const dealii::internal::EvaluatorTensorProduct<tensorproduct,
                                                       dim,
                                                       degree + 1,
                                                       n_points,
                                                       VNumber>
          eval_inv(dealii::AlignedVector<VNumber>(),
                   dealii::AlignedVector<VNumber>(),
                   data.get_matrix_free_x()
                     .get_shape_info()
                     .data[0]
                     .inverse_shape_values_eo);
        
        
        dealii::Tensor<1, dim, VectorizedArrayType> vel; // constant velocity
        
        
        {
            const unsigned int offset = 0;
            JxW              = &JxW_values[offset];
            JxW_inv          = &JxW_inv_values[offset];
            inverse_jacobian = &inverse_jacobian_values[offset];
        }

        // clang-format off

        // 1) advection: cell contribution
        {
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
                  Assert(false, dealii::StandardExceptions::ExcNotImplemented ());  
                }
            }

          // copy quadrature values into buffer
          VNumber *buffer = phi_cell_inv->get_data_ptr();
          
          if(eval_level != AdvectionOperationEvaluationLevel::cell)
          for (auto i = 0u; i < dealii::Utilities::pow<unsigned int>(n_points, dim); i++)
            buffer[i] = data_ptr[i];

          for(unsigned int d = 0; d < dim; ++d)
          {
            dealii::AlignedVector<VNumber> scratch_data_array;
            scratch_data_array.resize_fast(dealii::Utilities::pow(n_points, dim) );
            VNumber *tempp = scratch_data_array.begin();
            
            if(factor_skew != 0.0)
              {
                if (d == 0 && dim_x >= 1) eval_.template gradients<0, true, false>(buffer, tempp);
                if (d == 1 && dim_x >= 2) eval_.template gradients<1, true, false>(buffer, tempp);
                if (d == 2 && dim_x >= 3) eval_.template gradients<2, true, false>(buffer, tempp);
                if (d == 3 && dim_x >= 4) eval_.template gradients<3, true, false>(buffer, tempp);
                if (d == 4 && dim_x >= 5) eval_.template gradients<4, true, false>(buffer, tempp);
                if (d == 5 && dim_x >= 6) eval_.template gradients<5, true, false>(buffer, tempp);
              }
            
            for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
              for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                {
                  if(factor_skew != 0.0)
                    submit_value_cell<false>(data_ptr, - factor_skew * (get_gradient_cell(tempp, q, qx, qv, d) * vel[d]), q, qx, qv);
                  
                  if (factor_skew != 1.0)
                    submit_gradient_cell(tempp, (1.0-factor_skew) * buffer[q] * vel[d], q, qx, qv, d);
                }
            
            if(factor_skew != 1.0)
              {
                if (d == 0 && dim_x >= 1 && (factor_skew != 0.0))
                  eval_.template gradients<0, false, true >(tempp, data_ptr);
                else if (d == 0 && dim_x >= 1) 
                  eval_.template gradients<0, false, false>(tempp, data_ptr);
                  
                if (d == 1 && dim_x >= 2) eval_.template gradients<1, false, true >(tempp, data_ptr);
                if (d == 2 && dim_x >= 3) eval_.template gradients<2, false, true >(tempp, data_ptr);
                if (d == 3 && dim_x >= 4) eval_.template gradients<3, false, true >(tempp, data_ptr);
                if (d == 4 && dim_x >= 5) eval_.template gradients<4, false, true >(tempp, data_ptr);
                if (d == 5 && dim_x >= 6) eval_.template gradients<5, false, true >(tempp, data_ptr);
            }
          }
        }

        // 2) advection: faces
        if(eval_level != AdvectionOperationEvaluationLevel::cell)
        for (auto face = 0u; face < dim * 2; face++)
          {
            // load negative side from buffer
            phi_m.reinit(cell, face);
            
            {
              const unsigned int offset_face = (face) * dealii::Utilities::pow(n_points, dim_x + dim_v - 1);
              JxW_face = &JxW_face_values[offset_face];
            }
            
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
                dealii::internal::FEFaceNormalEvaluationImpl<dim, n_points - 1, 1, VectorizedArrayType, true>::template interpolate_quadrature<true, false>(data.get_matrix_free_x().get_shape_info(), /*out=*/data_ptr_inv, /*in=*/data_ptr1, false, face);
    
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
                    Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
                  }
              }
            else
              {
                Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
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
                          const VectorizedArrayType normal_times_speed          = vel[face/2] * ((face%2==0) ? -1.0 : 1.0 );
                          const VectorizedArrayType flux_times_normal_of_minus  = 0.5 * ((u_minus + u_plus) * normal_times_speed + std::abs(normal_times_speed) * (u_minus - u_plus)) * alpha;

                          submit_value_face<ID::SpaceType::X>(data_ptr1, flux_times_normal_of_minus - factor_skew*u_minus*normal_times_speed, q, qx, qv);
                        }
                  }
                else
                  {
                    for (unsigned int qv = 0, q = 0; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v - 1); ++qv)
                      for (unsigned int qx = 0; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
                        {
                          const VectorizedArrayType u_minus                     = data_ptr1[q];
                          const VectorizedArrayType u_plus                      = data_ptr2[q];
                          const VectorizedArrayType normal_times_speed          = vel[face/2] * ((face%2==0) ? -1.0 : 1.0 );
                          const VectorizedArrayType flux_times_normal_of_minus  = 0.5 * ((u_minus + u_plus) * normal_times_speed + std::abs(normal_times_speed) * (u_minus - u_plus)) * alpha;

                          submit_value_face<ID::SpaceType::V>(data_ptr1, flux_times_normal_of_minus - factor_skew*u_minus*normal_times_speed, q, qx, qv);
                        }
                  }
              }
            else
              {
                Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
              }

            if(do_collocation == false)
            {
              dealii::internal::FEFaceNormalEvaluationImpl<dim, n_points - 1, 1, VectorizedArrayType, true>::template interpolate_quadrature<false, true>(data.get_matrix_free_x().get_shape_info(), /*out=*/data_ptr1, /*in=*/data_ptr, false, face);
            }
            else
            {
                Assert(false, dealii::StandardExceptions::ExcNotImplemented ());
            }
            
          }

        // 3) inverse mass matrix
        {
          phi_inv.reinit(cell);

          for (auto qv = 0u, q = 0u; qv < dealii::Utilities::pow<unsigned int>(n_points, dim_v); ++qv)
            for (auto qx = 0u; qx < dealii::Utilities::pow<unsigned int>(n_points, dim_x); ++qx, ++q)
              submit_inv(data_ptr, q, qx, qv);

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
                  Assert(false, dealii::StandardExceptions::ExcNotImplemented ()); 
                }
            }

          // write into global structure back
          phi.set_dof_values(dst);
        }

        // clang-format on
      }
      
      template<bool do_add>
      void
      submit_value_cell(VectorizedArrayType *__restrict data_ptr_out,
                   const VectorizedArrayType __restrict values_in,
                   const unsigned int q,
                   const unsigned int qx,
                   const unsigned int qv)
      {
        (void) qx;
        (void) qv;
          
        const auto jxw = JxW_inv[q];

        if (do_add)
          data_ptr_out[q] += values_in * jxw;
        else
          data_ptr_out[q] = values_in * jxw;
      }
      
      void
      submit_inv(VectorizedArrayType *__restrict data_ptr_out,
                 const unsigned int q,
                 const unsigned int qx,
                 const unsigned int qv)
      {
        (void) qx;
        (void) qv;
          
        data_ptr_out[q] *= JxW_inv[q];
      }
      
      
    inline DEAL_II_ALWAYS_INLINE //
      VectorizedArrayType
      get_gradient_cell(const VectorizedArrayType *__restrict grad_in,
                     const unsigned int q,
                     const unsigned int qx,
                     const unsigned int qv, const unsigned int d)
    {
      (void) qx;
      (void) qv;
          
      return inverse_jacobian[0][d][d] * grad_in[q];
    }
    
      
      inline DEAL_II_ALWAYS_INLINE //
        void
        submit_gradient_cell(VectorizedArrayType *__restrict data_ptr_out,
                          const VectorizedArrayType &__restrict grad_in,
                          const unsigned int q,
                          const unsigned int qx,
                          const unsigned int qv, const unsigned int d)
      {
        (void) qx;
        (void) qv;
          
        data_ptr_out[q] = grad_in * JxW[q] * inverse_jacobian[0][d][d];
        
      }
      
      
    template <TensorID::SpaceType stype>
    inline DEAL_II_ALWAYS_INLINE //
      void
      submit_value_face(VectorizedArrayType *__restrict data_ptr,
                   const VectorizedArrayType &value,
                   const unsigned int         q,
                   const unsigned int         qx,
                   const unsigned int         qv)
    {
        (void) qx;
        (void) qv;
        
      data_ptr[q] = -value * JxW_face[q];
    }
    
      AlignedVector<typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX> JxW_values;
      AlignedVector<typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX> JxW_inv_values;
      AlignedVector<typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX> JxW_face_values;
      AlignedVector<Tensor<2,dim,typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX>> inverse_jacobian_values;
    
      typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX* JxW;
      typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX* JxW_inv;
      typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX* JxW_face;
      Tensor<2,dim,typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX>* inverse_jacobian;
      Tensor<1,dim,typename MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::VectorizedArrayTypeX>* normal;

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

      bool do_collocation;

      const double alpha = 1.0;

      // skew factor: conservative (skew=0) and convective (skew=1)
      double factor_skew = 0.0;
    };
    
  } // namespace advection
} // namespace hyperdeal

template <int dim_x, int dim_v>
struct Parameters
{
  Parameters(const std::string &               file_name,
             const dealii::ConditionalOStream &pcout)
    : n_subdivisions_x(dim_x, 0)
    , n_subdivisions_v(dim_v, 0)
  {
    dealii::ParameterHandler prm;

    std::ifstream file;
    file.open(file_name);

    add_parameters(prm);

    prm.parse_input_from_json(file, true);

    if (print_parameter && pcout.is_active())
      prm.print_parameters(pcout.get_stream(),
                           dealii::ParameterHandler::OutputStyle::Text);

    file.close();
  }

  void
  add_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("General");
    prm.add_parameter("Verbose", print_parameter);
    prm.add_parameter("Details", details);
    prm.add_parameter("Mapping", mapping);
    prm.leave_subsection();

    prm.enter_subsection("Performance");
    prm.add_parameter("Iterations", n_iterations);
    prm.add_parameter("IterationsWarmup", n_iterations_warmup);
    prm.leave_subsection();

    prm.enter_subsection("Case");

    prm.add_parameter("NRefinementsX", n_refinements_x);
    prm.add_parameter("NRefinementsV", n_refinements_v);

    prm.enter_subsection("NSubdivisionsX");
    if (dim_x >= 1)
      prm.add_parameter("X", n_subdivisions_x[0]);
    if (dim_x >= 2)
      prm.add_parameter("Y", n_subdivisions_x[1]);
    if (dim_x >= 3)
      prm.add_parameter("Z", n_subdivisions_x[2]);
    prm.leave_subsection();

    prm.enter_subsection("NSubdivisionsV");
    if (dim_v >= 1)
      prm.add_parameter("X", n_subdivisions_v[0]);
    if (dim_v >= 2)
      prm.add_parameter("Y", n_subdivisions_v[1]);
    if (dim_v >= 3)
      prm.add_parameter("Z", n_subdivisions_v[2]);
    prm.leave_subsection();

    prm.leave_subsection();
      
    prm.enter_subsection("AdvectionOperation");
    advection_operation_parameters.add_parameters(prm);
    prm.leave_subsection();
  }

  bool print_parameter = false;
  bool details         = false;

  unsigned int n_iterations_warmup = 0;
  unsigned int n_iterations        = 10;

  unsigned int n_refinements_x = 0;
  unsigned int n_refinements_v = 0;

  std::vector<unsigned int> n_subdivisions_x;
  std::vector<unsigned int> n_subdivisions_v;
    
  hyperdeal::advection::AdvectionOperationParamters advection_operation_parameters;
  
  std::string mapping;
};

template <int dim_x,
          int dim_v,
          int degree,
          int n_points,
          typename Number,
          typename VectorizedArrayType>
void
test(const MPI_Comm &                    comm_global,
     const MPI_Comm &                    comm_sm,
     const unsigned int                  size_x,
     const unsigned int                  size_v,
     hyperdeal::DynamicConvergenceTable &table,
     const std::string                   file_name)
{
  auto pcout = dealii::ConditionalOStream(
    std::cout, dealii::Utilities::MPI::this_mpi_process(comm_global) == 0);

  hyperdeal::MatrixFreeWrapper<dim_x, dim_v, Number, VectorizedArrayType>
    matrixfree_wrapper(comm_global, comm_sm, size_x, size_v);

  const Parameters<dim_x, dim_v> param(file_name, pcout);

  // clang-format off
  const dealii::Point< dim_x > px_1 = dim_x == 1 ? dealii::Point< dim_x >(0.0) : (dim_x == 2 ? dealii::Point< dim_x >(0.0, 0.0) : dealii::Point< dim_x >(0.0, 0.0, 0.0)); 
  const dealii::Point< dim_x > px_2 = dim_x == 1 ? dealii::Point< dim_x >(1.0) : (dim_x == 2 ? dealii::Point< dim_x >(1.0, 1.0) : dealii::Point< dim_x >(1.0, 1.0, 1.0)); 
  const dealii::Point< dim_v > pv_1 = dim_v == 1 ? dealii::Point< dim_v >(0.0) : (dim_v == 2 ? dealii::Point< dim_v >(0.0, 0.0) : dealii::Point< dim_v >(0.0, 0.0, 0.0)); 
  const dealii::Point< dim_v > pv_2 = dim_v == 1 ? dealii::Point< dim_v >(1.0) : (dim_v == 2 ? dealii::Point< dim_v >(1.0, 1.0) : dealii::Point< dim_v >(1.0, 1.0, 1.0));
  // clang-format on

  const bool do_periodic_x = true;
  const bool do_periodic_v = true;

  const hyperdeal::Parameters p(file_name, pcout);

  AssertThrow(p.degree == degree,
              dealii::StandardExceptions::ExcMessage(
                "Degrees " + std::to_string(p.degree) + " and " +
                std::to_string(degree) + " do not match!"));

  // clang-format off
  matrixfree_wrapper.init(p, [&](auto & tria_x, auto & tria_v){hyperdeal::GridGenerator::subdivided_hyper_rectangle(
    tria_x, tria_v,
    param.n_refinements_x, param.n_subdivisions_x, px_1, px_2, do_periodic_x, 
    param.n_refinements_v, param.n_subdivisions_v, pv_1, pv_2, do_periodic_v, /*deformation:*/ true);});
  // clang-format on

  const auto &matrix_free = matrixfree_wrapper.get_matrix_free();

  using VectorType = dealii::LinearAlgebra::SharedMPI::Vector<Number>;

  using VelocityFieldView =
    hyperdeal::advection::ConstantVelocityFieldView<dim_x + dim_v,
                                                    Number,
                                                    VectorizedArrayType,
                                                    dim_x,
                                                    dim_v>;

  auto boundary_descriptor = std::make_shared<
    hyperdeal::advection::BoundaryDescriptor<dim_x + dim_v, Number>>();

  auto velocity_field =
    std::make_shared<VelocityFieldView>(dealii::Tensor<1, dim_x + dim_v>());

  VectorType vec_src, vec_dst;
  matrix_free.initialize_dof_vector(vec_src, 0, true, true);
  matrix_free.initialize_dof_vector(vec_dst, 0, !p.use_ecl, true);

  hyperdeal::Timers timers(false);

  if(param.mapping == "full")
  {
     hyperdeal::advection::AdvectionOperation1<dim_x,
                                              dim_v,
                                              degree,
                                              n_points,
                                              Number,
                                              VectorType,
                                              VelocityFieldView,
                                              VectorizedArrayType>
       advection_operation(matrix_free, table);
   
     advection_operation.reinit(
       boundary_descriptor,
       velocity_field,
       param.advection_operation_parameters);
     
    timers.enter("apply");
    {
      hyperdeal::ScopedLikwidTimerWrapper likwid(
        std::string("apply") + "_" +
        std::to_string(dealii::Utilities::MPI::n_mpi_processes(comm_global)) +
        "_" + std::to_string(dealii::Utilities::MPI::n_mpi_processes(comm_sm)) +
        "_" + std::to_string(p.use_ecl) + "_" +
        std::to_string(VectorizedArrayType::size()) + "_" +
        std::to_string(degree) + "_" + std::to_string(dim_x + dim_v));

      timers.enter("withtimers");
      // run with timers
      hyperdeal::ScopedTimerWrapper timer(timers, "total");
      for (unsigned int i = 0; i < param.n_iterations; i++)
        advection_operation.apply(vec_dst, vec_src, 0.0, &timers);
      timers.leave();
    }
    timers.leave();
  }

  if(param.mapping == "cartesian")
  {
     hyperdeal::advection::AdvectionOperation2<dim_x,
                                              dim_v,
                                              degree,
                                              n_points,
                                              Number,
                                              VectorType,
                                              VelocityFieldView,
                                              VectorizedArrayType>
       advection_operation(matrix_free, table);
   
     advection_operation.reinit(
       boundary_descriptor,
       velocity_field,
       param.advection_operation_parameters);
     
    timers.enter("apply");
    {
      hyperdeal::ScopedLikwidTimerWrapper likwid(
        std::string("apply") + "_" +
        std::to_string(dealii::Utilities::MPI::n_mpi_processes(comm_global)) +
        "_" + std::to_string(dealii::Utilities::MPI::n_mpi_processes(comm_sm)) +
        "_" + std::to_string(p.use_ecl) + "_" +
        std::to_string(VectorizedArrayType::size()) + "_" +
        std::to_string(degree) + "_" + std::to_string(dim_x + dim_v));

      timers.enter("withtimers");
      // run with timers
      hyperdeal::ScopedTimerWrapper timer(timers, "total");
      for (unsigned int i = 0; i < param.n_iterations; i++)
        advection_operation.apply(vec_dst, vec_src, 0.0, &timers);
      timers.leave();
    }
    timers.leave();
  }

  table.set("info->size [DoFs]", matrixfree_wrapper.n_dofs());
  table.set(
    "info->ghost_size [DoFs]",
    Utilities::MPI::sum(matrix_free.get_vector_partitioner()->n_ghost_indices(),
                        comm_global));

  table.set("info->dim_x", dim_x);
  table.set("info->dim_v", dim_v);
  table.set("info->degree", degree);
  table.set("info->v_len", VectorizedArrayType::size());

  table.set("info->procs",
            dealii::Utilities::MPI::n_mpi_processes(comm_global));
  table.set("info->procs_x",
            dealii::Utilities::MPI::n_mpi_processes(
              matrixfree_wrapper.get_comm_row()));
  table.set("info->procs_v",
            dealii::Utilities::MPI::n_mpi_processes(
              matrixfree_wrapper.get_comm_column()));
  table.set("info->procs_sm", dealii::Utilities::MPI::n_mpi_processes(comm_sm));

  table.set("apply:total [s]",
            timers["apply:withtimers:total"].get_accumulated_time() / 1e6 /
              param.n_iterations);
  
  table.set("throughput - all1 [GDoFs/s]",
            matrixfree_wrapper.n_dofs() * param.n_iterations /
              timers["apply:withtimers:total"].get_accumulated_time() / 1000);

  std::vector<std::pair<std::string, std::string>> timer_labels;

  // clang-format off
  if(p.use_ecl)
  {
    timer_labels.emplace_back("apply:ECL:update_ghost_values_0", "apply:withtimers:ECL:update_ghost_values_0");
    timer_labels.emplace_back("apply:ECL:update_ghost_values_1", "apply:withtimers:ECL:update_ghost_values_1");
    timer_labels.emplace_back("apply:ECL:loop"               , "apply:withtimers:ECL:loop");
    timer_labels.emplace_back("apply:ECL:barrier"            , "apply:withtimers:ECL:barrier");
  }
  else
  {
    timer_labels.emplace_back("apply:FCL:advection"                     , "apply:withtimers:FCL:advection");
    timer_labels.emplace_back("apply:FCL:advection:update_ghost_values" , "apply:withtimers:FCL:advection:update_ghost_values");
    timer_labels.emplace_back("apply:FCL:advection:zero_out_ghosts"     , "apply:withtimers:FCL:advection:zero_out_ghosts");
    timer_labels.emplace_back("apply:FCL:advection:cell_loop"           , "apply:withtimers:FCL:advection:cell_loop");
    timer_labels.emplace_back("apply:FCL:advection:face_loop_x"         , "apply:withtimers:FCL:advection:face_loop_x");
    timer_labels.emplace_back("apply:FCL:advection:face_loop_v"         , "apply:withtimers:FCL:advection:face_loop_v");
    timer_labels.emplace_back("apply:FCL:advection:boundary_loop_x"     , "apply:withtimers:FCL:advection:boundary_loop_x");
    timer_labels.emplace_back("apply:FCL:advection:boundary_loop_v"     , "apply:withtimers:FCL:advection:boundary_loop_v");
    timer_labels.emplace_back("apply:FCL:advection:compress"            , "apply:withtimers:FCL:advection:compress");
    timer_labels.emplace_back("apply:FCL:mass"                          , "apply:withtimers:FCL:mass");
  }
  
  {
    std::vector<double> timing_values(timer_labels.size());
    
    for(unsigned int i = 0; i < timer_labels.size(); i++)
      timing_values[i] = timers[timer_labels[i].second].get_accumulated_time() / 1e6 / param.n_iterations;
   
    const auto timing_values_min_max_avg = Utilities::MPI::min_max_avg (timing_values, comm_global);
    
    for(unsigned int i = 0; i < timer_labels.size(); i++)
    {
        const auto min_max_avg = timing_values_min_max_avg[i];
        table.set(timer_labels[i].first + ":avg [s]", min_max_avg.avg);
        table.set(timer_labels[i].first + ":min [s]", min_max_avg.min);
        table.set(timer_labels[i].first + ":max [s]", min_max_avg.max);
    }
  }

  // clang-format on
}
