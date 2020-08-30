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

#ifndef HYPERDEAL_NDIM_OPERATORS_POISSON
#define HYPERDEAL_NDIM_OPERATORS_POISSON

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

template <int dim_,
          int fe_degree,
          int n_q_points_1d,
          typename Number,
          typename VectorizedArrayType_>
class LaplaceOperator : public dealii::Subscriptor
{
public:
  static const int dim      = dim_;
  using value_type          = Number;
  using VectorizedArrayType = VectorizedArrayType_;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  void
  initialize(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const bool do_zero_mean = false)
  {
    this->matrix_free  = &matrix_free;
    this->do_zero_mean = do_zero_mean;
  }

  const dealii::MatrixFree<dim, Number, VectorizedArrayType> &
  get_matrix_free() const
  {
    return *this->matrix_free;
  }

  void
  initialize_dof_vector(VectorType &vector) const
  {
    matrix_free->initialize_dof_vector(vector);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    VectorType        src_;
    const VectorType *src_ptr = &src;

    if (do_zero_mean)
      {
        src_.reinit(src, true);
        src_.copy_locally_owned_data_from(src);

        src_.add(-src_.mean_value());

        src_ptr = &src_;
      }


    matrix_free->loop(&LaplaceOperator::local_apply,
                      &LaplaceOperator::local_apply_face,
                      &LaplaceOperator::local_apply_boundary,
                      this,
                      dst,
                      *src_ptr);
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    this->vmult(dst, src);
  }

  void
  compute_inverse_diagonal(VectorType &inverse_diagonal_entries) const
  {
    matrix_free->initialize_dof_vector(inverse_diagonal_entries);
    inverse_diagonal_entries = 0.0;
    unsigned int dummy;
    matrix_free->loop(&LaplaceOperator::local_diagonal_cell,
                      &LaplaceOperator::local_diagonal_face,
                      &LaplaceOperator::local_diagonal_boundary,
                      this,
                      inverse_diagonal_entries,
                      dummy);

    for (unsigned int i = 0; i < inverse_diagonal_entries.local_size(); ++i)
      if (std::abs(inverse_diagonal_entries.local_element(i)) > 1e-10)
        inverse_diagonal_entries.local_element(i) =
          1. / inverse_diagonal_entries.local_element(i);
  }

  Number
  el(const dealii::types::global_dof_index,
     const dealii::types::global_dof_index) const
  {
    AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());

    return 0;
  }

  dealii::types::global_dof_index
  m() const
  {
    return matrix_free->get_vector_partitioner()->size();
  }

private:
  void
  local_apply(const dealii::MatrixFree<dim, Number, VectorizedArrayType> &data,
              dealii::LinearAlgebra::distributed::Vector<Number> &        dst,
              const dealii::LinearAlgebra::distributed::Vector<Number> &  src,
              const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    dealii::FEEvaluation<dim,
                         fe_degree,
                         n_q_points_1d,
                         1,
                         Number,
                         VectorizedArrayType>
      phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        phi.evaluate(false, true, false);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_gradient(phi.get_gradient(q), q);
        phi.integrate(false, true);
        phi.set_dof_values(dst);
      }
  }

  void
  local_apply_face(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> &data,
    dealii::LinearAlgebra::distributed::Vector<Number> &        dst,
    const dealii::LinearAlgebra::distributed::Vector<Number> &  src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    dealii::FEFaceEvaluation<dim,
                             fe_degree,
                             n_q_points_1d,
                             1,
                             Number,
                             VectorizedArrayType>
      fe_eval(data, true);
    dealii::FEFaceEvaluation<dim,
                             fe_degree,
                             n_q_points_1d,
                             1,
                             Number,
                             VectorizedArrayType>
      fe_eval_neighbor(data, false);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval.reinit(face);
        fe_eval_neighbor.reinit(face);

        fe_eval.read_dof_values(src);
        fe_eval.evaluate(true, true);
        fe_eval_neighbor.read_dof_values(src);
        fe_eval_neighbor.evaluate(true, true);
        VectorizedArrayType sigmaF =
          (std::abs((fe_eval.get_normal_vector(0) *
                     fe_eval.inverse_jacobian(0))[dim - 1]) +
           std::abs((fe_eval.get_normal_vector(0) *
                     fe_eval_neighbor.inverse_jacobian(0))[dim - 1])) *
          (Number)(std::max(fe_degree, 1) * (fe_degree + 1.0));

        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          {
            VectorizedArrayType average_value =
              (fe_eval.get_value(q) - fe_eval_neighbor.get_value(q)) * 0.5;
            VectorizedArrayType average_valgrad =
              fe_eval.get_normal_derivative(q) +
              fe_eval_neighbor.get_normal_derivative(q);
            average_valgrad =
              average_value * 2. * sigmaF - average_valgrad * 0.5;
            fe_eval.submit_normal_derivative(-average_value, q);
            fe_eval_neighbor.submit_normal_derivative(-average_value, q);
            fe_eval.submit_value(average_valgrad, q);
            fe_eval_neighbor.submit_value(-average_valgrad, q);
          }
        fe_eval.integrate(true, true);
        fe_eval.distribute_local_to_global(dst);
        fe_eval_neighbor.integrate(true, true);
        fe_eval_neighbor.distribute_local_to_global(dst);
      }
  }

  void
  local_apply_boundary(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> &data,
    dealii::LinearAlgebra::distributed::Vector<Number> &        dst,
    const dealii::LinearAlgebra::distributed::Vector<Number> &  src,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    dealii::FEFaceEvaluation<dim,
                             fe_degree,
                             n_q_points_1d,
                             1,
                             Number,
                             VectorizedArrayType>
      fe_eval(data, true);
    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        fe_eval.reinit(face);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(true, true);
        VectorizedArrayType sigmaF =
          std::abs((fe_eval.get_normal_vector(0) *
                    fe_eval.inverse_jacobian(0))[dim - 1]) *
          Number(std::max(fe_degree, 1) * (fe_degree + 1.0)) * 2.0;

        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          {
            VectorizedArrayType average_value = fe_eval.get_value(q);
            VectorizedArrayType average_valgrad =
              -fe_eval.get_normal_derivative(q);
            average_valgrad += average_value * sigmaF * 2.0;
            fe_eval.submit_normal_derivative(-average_value, q);
            fe_eval.submit_value(average_valgrad, q);
          }

        fe_eval.integrate(true, true);
        fe_eval.distribute_local_to_global(dst);
      }
  }

  void
  local_diagonal_cell(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> &data,
    dealii::LinearAlgebra::distributed::Vector<Number> &        dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    dealii::FEEvaluation<dim,
                         fe_degree,
                         n_q_points_1d,
                         1,
                         Number,
                         VectorizedArrayType>
      phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);

        VectorizedArrayType local_diagonal_vector[phi.static_dofs_per_cell];
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              phi.begin_dof_values()[j] = VectorizedArrayType();
            phi.begin_dof_values()[i] = 1.;
            phi.evaluate(false, true, false);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_gradient(phi.get_gradient(q), q);
            phi.integrate(false, true);
            local_diagonal_vector[i] = phi.begin_dof_values()[i];
          }
        for (unsigned int i = 0; i < phi.static_dofs_per_cell; ++i)
          phi.begin_dof_values()[i] = local_diagonal_vector[i];
        phi.set_dof_values(dst);
      }
  }

  void
  local_diagonal_face(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> &data,
    dealii::LinearAlgebra::distributed::Vector<Number> &        dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    dealii::FEFaceEvaluation<dim,
                             fe_degree,
                             n_q_points_1d,
                             1,
                             Number,
                             VectorizedArrayType>
      phi(data, true);
    dealii::FEFaceEvaluation<dim,
                             fe_degree,
                             n_q_points_1d,
                             1,
                             Number,
                             VectorizedArrayType>
      phi_outer(data, false);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        phi.reinit(face);
        phi_outer.reinit(face);

        VectorizedArrayType local_diagonal_vector[phi.static_dofs_per_cell];
        VectorizedArrayType sigmaF =
          (std::abs(
             (phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]) +
           std::abs((phi.get_normal_vector(0) *
                     phi_outer.inverse_jacobian(0))[dim - 1])) *
          (Number)(std::max(fe_degree, 1) * (fe_degree + 1.0));

        // Compute phi part
        for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
          phi_outer.begin_dof_values()[j] = VectorizedArrayType();
        phi_outer.evaluate(true, true);
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              phi.begin_dof_values()[j] = VectorizedArrayType();
            phi.begin_dof_values()[i] = 1.;
            phi.evaluate(true, true);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                VectorizedArrayType average_value =
                  (phi.get_value(q) - phi_outer.get_value(q)) * 0.5;
                VectorizedArrayType average_valgrad =
                  phi.get_normal_derivative(q) +
                  phi_outer.get_normal_derivative(q);
                average_valgrad =
                  average_value * 2. * sigmaF - average_valgrad * 0.5;
                phi.submit_normal_derivative(-average_value, q);
                phi.submit_value(average_valgrad, q);
              }
            phi.integrate(true, true);
            local_diagonal_vector[i] = phi.begin_dof_values()[i];
          }
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          phi.begin_dof_values()[i] = local_diagonal_vector[i];
        phi.distribute_local_to_global(dst);

        // Compute phi_outer part
        for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
          phi.begin_dof_values()[j] = VectorizedArrayType();
        phi.evaluate(true, true);
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              phi_outer.begin_dof_values()[j] = VectorizedArrayType();
            phi_outer.begin_dof_values()[i] = 1.;
            phi_outer.evaluate(true, true);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                VectorizedArrayType average_value =
                  (phi.get_value(q) - phi_outer.get_value(q)) * 0.5;
                VectorizedArrayType average_valgrad =
                  phi.get_normal_derivative(q) +
                  phi_outer.get_normal_derivative(q);
                average_valgrad =
                  average_value * 2. * sigmaF - average_valgrad * 0.5;
                phi_outer.submit_normal_derivative(-average_value, q);
                phi_outer.submit_value(-average_valgrad, q);
              }
            phi_outer.integrate(true, true);
            local_diagonal_vector[i] = phi_outer.begin_dof_values()[i];
          }
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          phi_outer.begin_dof_values()[i] = local_diagonal_vector[i];
        phi_outer.distribute_local_to_global(dst);
      }
  }

  void
  local_diagonal_boundary(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> &data,
    dealii::LinearAlgebra::distributed::Vector<Number> &        dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &face_range) const
  {
    dealii::FEFaceEvaluation<dim,
                             fe_degree,
                             n_q_points_1d,
                             1,
                             Number,
                             VectorizedArrayType>
      phi(data);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        phi.reinit(face);

        VectorizedArrayType local_diagonal_vector[phi.static_dofs_per_cell];
        VectorizedArrayType sigmaF =
          std::abs(
            (phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]) *
          Number(std::max(fe_degree, 1) * (fe_degree + 1.0)) * 2.0;

        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              phi.begin_dof_values()[j] = VectorizedArrayType();
            phi.begin_dof_values()[i] = 1.;
            phi.evaluate(true, true);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                VectorizedArrayType average_value = phi.get_value(q);
                VectorizedArrayType average_valgrad =
                  -phi.get_normal_derivative(q);
                average_valgrad += average_value * sigmaF * 2.0;
                phi.submit_normal_derivative(-average_value, q);
                phi.submit_value(average_valgrad, q);
              }

            phi.integrate(true, true);
            local_diagonal_vector[i] = phi.begin_dof_values()[i];
          }
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          phi.begin_dof_values()[i] = local_diagonal_vector[i];
        phi.distribute_local_to_global(dst);
      }
  }

  const dealii::MatrixFree<dim, Number, VectorizedArrayType> *matrix_free;

  bool do_zero_mean;
};

template <int dim, typename LAPLACEOPERATOR>
class MGTransferMF
  : public dealii::MGTransferMatrixFree<dim,
                                        typename LAPLACEOPERATOR::value_type>
{
public:
  MGTransferMF(const dealii::MGLevelObject<LAPLACEOPERATOR> &laplace,
               const dealii::MGConstrainedDoFs &mg_constrained_dofs)
    : dealii::MGTransferMatrixFree<dim, typename LAPLACEOPERATOR::value_type>(
        mg_constrained_dofs)
    , laplace_operator(laplace){};

  /**
   * Overload copy_to_mg from MGTransferPrebuilt
   */
  template <class InVector, int spacedim>
  void
  copy_to_mg(const dealii::DoFHandler<dim, spacedim> & mg_dof_handler,
             dealii::MGLevelObject<dealii::LinearAlgebra::distributed::Vector<
               typename LAPLACEOPERATOR::value_type>> &dst,
             const InVector &                          src) const
  {
    for (unsigned int level = dst.min_level(); level <= dst.max_level();
         ++level)
      laplace_operator[level].initialize_dof_vector(dst[level]);
    dealii::MGTransferMatrixFree<dim, typename LAPLACEOPERATOR::value_type>::
      copy_to_mg(mg_dof_handler, dst, src);
  }

private:
  const dealii::MGLevelObject<LAPLACEOPERATOR> &laplace_operator;
};

template <typename VectorType>
class PoissonSolverBase
{
public:
  virtual unsigned int
  solve(VectorType &dst, const VectorType &src) = 0;
};

template <typename MatrixType>
class PoissonSolver : public PoissonSolverBase<typename MatrixType::VectorType>
{
public:
  using VectorType                 = typename MatrixType::VectorType;
  using SmootherPreconditionerType = dealii::DiagonalMatrix<VectorType>;
  using SmootherType               = dealii::
    PreconditionChebyshev<MatrixType, VectorType, SmootherPreconditionerType>;


  PoissonSolver(const MatrixType &fine_matrix)
    : fine_matrix(fine_matrix)
  {
    typename SmootherType::AdditionalData ad;
    ad.preconditioner = std::make_shared<SmootherPreconditionerType>();
    fine_matrix.compute_inverse_diagonal(ad.preconditioner->get_vector());
    preconditioner.initialize(fine_matrix, ad);
  }

  unsigned int
  solve(VectorType &dst, const VectorType &src) override
  {
    dealii::ReductionControl     solver_control(src.size() * 2,
                                            1e-20,
                                            1e-7); // [TODO]
    dealii::SolverCG<VectorType> solver(solver_control);


    solver.solve(fine_matrix, dst, src, preconditioner);

    return solver_control.last_step();
  }

private:
  const MatrixType &fine_matrix;
  SmootherType      preconditioner;
};

template <typename MatrixType>
class PoissonSolverMG
  : public PoissonSolverBase<typename MatrixType::VectorType>
{
public:
  static const int dim      = MatrixType::dim;
  using Number              = typename MatrixType::value_type;
  using VectorizedArrayType = typename MatrixType::VectorizedArrayType;
  using MatrixFreeType = dealii::MatrixFree<dim, Number, VectorizedArrayType>;
  using VectorType     = typename MatrixType::VectorType;
  using SmootherPreconditionerType = dealii::DiagonalMatrix<VectorType>;
  using SmootherType               = dealii::
    PreconditionChebyshev<MatrixType, VectorType, SmootherPreconditionerType>;
  using MGTransferType = MGTransferMF<dim, MatrixType>;

  using PreconditionerType =
    dealii::PreconditionMG<dim, VectorType, MGTransferType>;


  PoissonSolverMG(
    const MatrixType &fine_matrix,
    const std::shared_ptr<dealii::MGLevelObject<MatrixFreeType>>
                                                             level_matrix_free,
    const std::shared_ptr<dealii::MGLevelObject<MatrixType>> mg_matrices_)
    : fine_matrix(fine_matrix)
    , level_matrix_free(level_matrix_free)
    , mg_matrices_(mg_matrices_)
    , min_level(mg_matrices_->min_level())
    , max_level(mg_matrices_->max_level())
  {
    const auto &dof = fine_matrix.get_matrix_free().get_dof_handler();

    dealii::MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
      min_level, max_level);

    const auto &mg_matrices = *mg_matrices_;

    // initialize levels
    for (unsigned int level = min_level; level <= max_level; level++)
      {
        // ... initialize smoother
        smoother_data[level].preconditioner =
          std::make_shared<SmootherPreconditionerType>();
        mg_matrices[level].compute_inverse_diagonal(
          smoother_data[level].preconditioner->get_vector());
        smoother_data[level].smoothing_range     = 20.; // [TODO]
        smoother_data[level].degree              = 5;   // [TODO]
        smoother_data[level].eig_cg_n_iterations = 15;  // [TODO]

        // ... use coarse smoother as preconditioner on coarsest level
        if (level == min_level)
          mg_coarse_grid_smoother.initialize(mg_matrices[level],
                                             smoother_data[level]);
      }

    // ... initialize level mg_matrices
    mg_matrix.initialize(mg_matrices);

    // ... initialize level smoothers
    mg_smoother.initialize(mg_matrices, smoother_data);

    // initialize coarse-grid solver
    coarse_grid_solver_control = std::make_shared<dealii::ReductionControl>(
      1e4, 1e-12, 1e-3, false, false); // [TODO]
    coarse_grid_solver = std::make_shared<dealii::SolverCG<VectorType>>(
      *coarse_grid_solver_control);

    mg_coarse.initialize(*coarse_grid_solver,
                         mg_matrices[min_level],
                         mg_coarse_grid_smoother);

    // initialize transfer operator
    mg_constrained_dofs.initialize(dof);

    mg_transfer =
      std::make_shared<MGTransferType>(mg_matrices, mg_constrained_dofs);
    mg_transfer->build(dof);

    // create multigrid object
    mg = std::make_shared<dealii::Multigrid<VectorType>>(
      mg_matrix, mg_coarse, *mg_transfer, mg_smoother, mg_smoother);

    // convert it to a precondtioner
    preconditioner =
      std::make_shared<PreconditionerType>(dof, *mg, *mg_transfer);
  }

  unsigned int
  solve(VectorType &dst, const VectorType &src) override
  {
    dealii::ReductionControl     solver_control(src.size() * 2,
                                            1e-20,
                                            1e-4); // [TODO]
    dealii::SolverCG<VectorType> solver(solver_control);

    solver.solve(fine_matrix, dst, src, *preconditioner);

    return solver_control.last_step();
  }

private:
  const MatrixType &fine_matrix;
  const std::shared_ptr<dealii::MGLevelObject<MatrixFreeType>>
                                                           level_matrix_free;
  const std::shared_ptr<dealii::MGLevelObject<MatrixType>> mg_matrices_;

  const unsigned int min_level;
  const unsigned int max_level;

  SmootherType mg_coarse_grid_smoother;

  dealii::MGSmootherPrecondition<MatrixType, SmootherType, VectorType>
                                 mg_smoother;
  dealii::mg::Matrix<VectorType> mg_matrix;

  std::shared_ptr<dealii::ReductionControl> coarse_grid_solver_control;

  std::shared_ptr<dealii::SolverCG<VectorType>> coarse_grid_solver;

  dealii::MGCoarseGridIterativeSolver<VectorType,
                                      dealii::SolverCG<VectorType>,
                                      MatrixType,
                                      SmootherType>
    mg_coarse;

  dealii::MGConstrainedDoFs       mg_constrained_dofs;
  std::shared_ptr<MGTransferType> mg_transfer;

  std::shared_ptr<dealii::Multigrid<VectorType>> mg;
  std::shared_ptr<PreconditionerType>            preconditioner;
};

#endif
