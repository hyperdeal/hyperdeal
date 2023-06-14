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


// Test FEFaceEvaluation::read_dof_values() for ECL/FCL and for different
// orientations.

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <hyper.deal/base/mpi.h>
#include <hyper.deal/base/utilities.h>
#include <hyper.deal/grid/grid_generator.h>
#include <hyper.deal/matrix_free/evaluation_kernels.h>
#include <hyper.deal/matrix_free/fe_evaluation_face.h>
#include <hyper.deal/numerics/vector_tools.h>

#include <ios>

#include "../tests_functions.h"
#include "../tests_mf.h"

using namespace dealii;

template <int DIM, typename Number = double>
class ExactSolution : public dealii::Function<DIM, Number>
{
public:
  ExactSolution(const double direction)
    : dealii::Function<DIM, Number>(1)
    , direction(direction)
  {}

  virtual double
  value(const dealii::Point<DIM> &p, const unsigned int = 1) const
  {
    return p[direction];
  }

private:
  const unsigned int direction;
};

template <
  int dim_x,
  int dim_v,
  int degree,
  int n_points,
  typename Number              = double,
  typename VectorizedArrayType = dealii::VectorizedArray<double, 1>,
  typename VectorType = dealii::LinearAlgebra::distributed::Vector<double>>
void
test(const MPI_Comm &comm)
{
  const auto sizes = hyperdeal::Utilities::decompose(
    dealii::Utilities::MPI::n_mpi_processes(comm));

  const unsigned int size_x = sizes.first;
  const unsigned int size_v = sizes.second;

  MPI_Comm comm_global =
    hyperdeal::mpi::create_rectangular_comm(comm, size_x, size_v);


  if (comm_global != MPI_COMM_NULL)
    {
      MPI_Comm comm_sm = MPI_COMM_SELF;

      hyperdeal::MatrixFreeWrapper<dim_x, dim_v, Number, VectorizedArrayType>
        matrixfree_wrapper(comm_global, comm_sm, size_x, size_v);

      hyperdeal::Parameters p;
      p.triangulation_type = "fullydistributed";
      p.degree             = degree;
      p.mapping_degree     = 1;
      p.do_collocation     = false;
      p.do_ghost_faces     = true;
      p.do_buffering       = false;
      p.use_ecl            = true;

      matrixfree_wrapper.init(p, [&](auto &tria_x, auto &tria_v) {
        hyperdeal::GridGenerator::hyper_cube(tria_x, tria_v, false, 0);
      });

      const auto &matrix_free = matrixfree_wrapper.get_matrix_free();

      hyperdeal::FEEvaluation<dim_x,
                              dim_v,
                              degree,
                              n_points,
                              Number,
                              VectorizedArrayType>
        phi(matrix_free, 0, 0, 0, 0);


      hyperdeal::FEEvaluation<dim_x,
                              dim_v,
                              degree,
                              n_points,
                              Number,
                              VectorizedArrayType>
        phi_cell_inv(matrix_free, 0, 0, 0, 0);

      hyperdeal::FEFaceEvaluation<dim_x,
                                  dim_v,
                                  degree,
                                  n_points,
                                  Number,
                                  VectorizedArrayType>
        phi_m(matrix_free, true, 0, 0, 0, 0);


      static const int dim = dim_x + dim_v;

      deallog.push("dim_x=" + std::to_string(dim_x) + ":" +
                   std::to_string(dim_v));

      for (int direction = 0; direction < dim; ++direction)
        {
          deallog.push("dir=" + std::to_string(direction));

          VectorType vec;
          matrix_free.initialize_dof_vector(vec, 0, true, true);

          std::shared_ptr<dealii::Function<dim, Number>> solution(
            new ExactSolution<dim, Number>(direction));

          hyperdeal::VectorTools::interpolate<degree, n_points>(
            solution, matrix_free, vec, 0, 0, 2, 2);

          deallog.push("FCL");

          matrix_free.template loop<VectorType, VectorType>(
            [&](const auto &, auto &, const auto &, const auto) {},
            [&](const auto &, auto &, const auto &, const auto) {

            },
            [&](const auto &, auto &, const auto &src, const auto face) {
              deallog.push("dir=" + std::to_string(face.macro));

              phi_m.reinit(face);
              phi_m.read_dof_values(src);

              VectorizedArrayType *data_ptr1 = phi_m.get_data_ptr();

              dealii::internal::FEEvaluationImplBasisChange<
                dealii::internal::EvaluatorVariant::evaluate_evenodd,
                dealii::internal::EvaluatorQuantity::value,
                dim - 1,
                degree + 1,
                n_points,
                VectorizedArrayType,
                VectorizedArrayType>::do_forward(1,
                                                 matrix_free.get_matrix_free_x()
                                                   .get_shape_info()
                                                   .data.front()
                                                   .shape_values_eo,
                                                 data_ptr1,
                                                 data_ptr1);

              std::set<unsigned int> failed;

              for (unsigned int q = 0;
                   q < dealii::Utilities::pow<unsigned int>(n_points, dim - 1);
                   ++q)
                {
                  if (std::abs(phi_m.get_quadrature_point(q)[direction][0] -
                               phi_m.get_data_ptr()[q][0]) > 1e-8)
                    failed.insert(q);
                }

              if (failed.empty())
                deallog << "succeeded!" << std::endl;
              else
                {
                  deallog << " failed!" << std::endl;
                }

              deallog.pop();
            },
            vec,
            vec,
            hyperdeal::MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
              DataAccessOnFaces::values,
            hyperdeal::MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
              DataAccessOnFaces::values);

          deallog.pop();

          deallog.push("ECL");

          matrix_free.template loop_cell_centric<VectorType, VectorType>(
            [&](const auto &, auto &, const auto &src, const auto cell) {
              phi.reinit(cell);
              phi.read_dof_values(src);

              VectorizedArrayType *data_ptr  = phi.get_data_ptr();
              VectorizedArrayType *data_ptr1 = phi_m.get_data_ptr();

              dealii::internal::FEEvaluationImplBasisChange<
                dealii::internal::EvaluatorVariant::evaluate_evenodd,
                dealii::internal::EvaluatorQuantity::value,
                dim,
                degree + 1,
                n_points,
                VectorizedArrayType,
                VectorizedArrayType>::do_forward(1,
                                                 matrix_free.get_matrix_free_x()
                                                   .get_shape_info()
                                                   .data.front()
                                                   .shape_values_eo,
                                                 data_ptr,
                                                 data_ptr);


              VectorizedArrayType *buffer = phi_cell_inv.get_data_ptr();

              for (auto i = 0u;
                   i < dealii::Utilities::pow<unsigned int>(n_points, dim);
                   i++)
                buffer[i] = data_ptr[i];

              for (auto face = 0u; face < dim * 2; face++)
                {
                  deallog.push("dir=" + std::to_string(face));

                  phi_m.reinit(cell, face);


                  hyperdeal::internal::FEFaceNormalEvaluationImpl<
                    dim_x,
                    dim_v,
                    n_points - 1,
                    VectorizedArrayType>::
                    template interpolate_quadrature<true, false>(
                      1,
                      dealii::EvaluationFlags::values,
                      matrix_free.get_matrix_free_x().get_shape_info(),
                      /*out=*/phi_cell_inv.get_data_ptr(),
                      /*in=*/data_ptr1,
                      face);

                  std::set<unsigned int> failed;

                  for (unsigned int q = 0;
                       q <
                       dealii::Utilities::pow<unsigned int>(n_points, dim - 1);
                       ++q)
                    {
                      if (std::abs(phi_m.get_quadrature_point(q)[direction][0] -
                                   phi_m.get_data_ptr()[q][0]) > 1e-8)
                        failed.insert(q);
                    }

                  if (failed.empty())
                    deallog << "succeeded!" << std::endl;
                  else
                    {
                      deallog << "failed!" << std::endl;
                    }

                  deallog.pop();
                }
            },
            vec,
            vec,
            hyperdeal::MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
              DataAccessOnFaces::values);

          deallog.pop();
          deallog.pop();
        }

      deallog.pop();

      MPI_Comm_free(&comm_global);
    }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  test<1, 1, 3, 4>(comm);

  test<2, 1, 3, 4>(comm);
  test<1, 2, 3, 4>(comm);

  test<1, 3, 3, 4>(comm);
  test<2, 2, 3, 4>(comm);
  test<3, 1, 3, 4>(comm);

  test<2, 3, 3, 4>(comm);
  test<3, 2, 3, 4>(comm);

  test<3, 3, 3, 4>(comm);
}
