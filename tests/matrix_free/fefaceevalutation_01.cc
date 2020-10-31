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

#include <hyper.deal/base/mpi.h>
#include <hyper.deal/base/utilities.h>
#include <hyper.deal/grid/grid_generator.h>
#include <hyper.deal/lac/sm_vector.h>
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
  ExactSolution(const double time = 0.)
    : dealii::Function<DIM, Number>(1, time)
    , wave_number(2.)
  {
    advection[0] = 1. * 0.0;
    if (DIM > 1)
      advection[1] = 0.15 * 0.0;
    if (DIM > 2)
      advection[2] = -0.05 * 0.0;
    if (DIM > 3)
      advection[3] = -0.10;
    if (DIM > 4)
      advection[4] = -0.15;
    if (DIM > 5)
      advection[5] = 0.5;
  }

  virtual double
  value(const dealii::Point<DIM> &p, const unsigned int = 1) const
  {
    double                       t        = this->get_time();
    const dealii::Tensor<1, DIM> position = p - t * advection;
    double result = std::sin(wave_number * position[0] * dealii::numbers::PI);
    for (unsigned int d = 1; d < DIM; ++d)
      result *= std::cos(wave_number * position[d] * dealii::numbers::PI);
    return result;
  }

  dealii::Tensor<1, DIM>
  get_transport_direction() const
  {
    return advection;
  }

private:
  dealii::Tensor<1, DIM> advection;
  const double           wave_number;
};

template <int dim_x,
          int dim_v,
          int degree,
          int n_points,
          typename Number,
          typename VectorizedArrayType,
          typename VectorType>
void
test(const MPI_Comm &comm, const unsigned int o1, const unsigned int o2)
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
        dealii::Point<dim_x> left_x(-1, -1, -1);
        dealii::Point<dim_x> right_x(+1, +1, +1);
        dealii::Point<dim_v> left_v(-1, -1, -1);
        dealii::Point<dim_v> right_v(+1, +1, +1);

        hyperdeal::GridGenerator::orientated_hyper_cube(tria_x,
                                                        tria_v,
                                                        0,
                                                        left_x,
                                                        right_x,
                                                        true,
                                                        o1,
                                                        0,
                                                        left_v,
                                                        right_v,
                                                        true,
                                                        o2);
      });

      const auto &matrix_free = matrixfree_wrapper.get_matrix_free();

      VectorType vec;
      matrix_free.initialize_dof_vector(vec, 0, true, true);

      static const int dim = dim_x + dim_v;

      std::shared_ptr<dealii::Function<dim, Number>> solution(
        new ExactSolution<dim, Number>);

      hyperdeal::VectorTools::interpolate<degree, n_points>(
        solution, matrix_free, vec, 0, 0, 2, 2);

      hyperdeal::FEEvaluation<dim_x,
                              dim_v,
                              degree,
                              n_points,
                              Number,
                              VectorizedArrayType>
        phi(matrix_free, 0, 0, 0, 0);
      hyperdeal::FEFaceEvaluation<dim_x,
                                  dim_v,
                                  degree,
                                  n_points,
                                  Number,
                                  VectorizedArrayType>
        phi_m(matrix_free, true, 0, 0, 0, 0);
      hyperdeal::FEFaceEvaluation<dim_x,
                                  dim_v,
                                  degree,
                                  n_points,
                                  Number,
                                  VectorizedArrayType>
        phi_p(matrix_free, false, 0, 0, 0, 0);

      {
        deallog << "FCL - Orientation (" << o1 << "," << o2 << "): ";

        std::set<unsigned int> failed;

        matrix_free.template loop<VectorType, VectorType>(
          [&](const auto &, auto &, const auto &, const auto) {},
          [&](const auto &, auto &, const auto &src, const auto face) {
            phi_m.reinit(face);
            phi_m.read_dof_values(src);

            phi_p.reinit(face);
            phi_p.read_dof_values(src);

            for (unsigned int q = 0;
                 q < dealii::Utilities::pow<unsigned int>(n_points, dim - 1);
                 ++q)
              {
                if (std::abs(phi_p.get_data_ptr()[q] -
                             phi_m.get_data_ptr()[q])[0] > 1e-8)
                  failed.insert(face.macro);
              }
          },
          [&](const auto &, auto &, const auto &, const auto) {},
          vec,
          vec,
          hyperdeal::MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
            DataAccessOnFaces::values,
          hyperdeal::MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
            DataAccessOnFaces::values);

        if (failed.empty())
          deallog << "succeeded!" << std::endl;
        else
          {
            deallog << " failed!" << std::endl;
          }
      }

      {
        deallog << "ECL - Orientation (" << o1 << "," << o2 << "): ";

        std::set<std::pair<unsigned int, unsigned int>> failed;

        matrix_free.template loop_cell_centric<VectorType, VectorType>(
          [&](const auto &, auto &, const auto &src, const auto cell) {
            phi.reinit(cell);
            phi.read_dof_values(src);


            for (auto face = 0u; face < dim * 2; face++)
              {
                phi_m.reinit(cell, face);
                phi_m.read_dof_values(src);

                phi_p.reinit(cell, face);
                phi_p.read_dof_values(src);

                for (unsigned int q = 0;
                     q <
                     dealii::Utilities::pow<unsigned int>(n_points, dim - 1);
                     ++q)
                  {
                    if (std::abs(phi_p.get_data_ptr()[q] -
                                 phi.get_data_ptr()
                                   [matrix_free.get_shape_info()
                                      .face_to_cell_index_nodal[face][q]])[0] >
                          1e-8 &&
                        std::abs(phi_p.get_data_ptr()[q] -
                                 phi_m.get_data_ptr()[q])[0] > 1e-8)
                      failed.insert({cell.macro, face});
                  }
              }
          },
          vec,
          vec,
          hyperdeal::MatrixFree<dim_x, dim_v, Number, VectorizedArrayType>::
            DataAccessOnFaces::values);

        if (failed.empty())
          deallog << "succeeded!" << std::endl;
        else
          {
            for (auto i : failed)
              deallog << i.first << "/" << i.second << " ";

            deallog << " failed!" << std::endl;
          }
      }

      MPI_Comm_free(&comm_global);
    }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  for (unsigned int i = 0; i < 16; ++i)
    for (unsigned int j = 0; j < 16; ++j)
      test<3,
           3,
           3,
           4,
           double,
           dealii::VectorizedArray<double, 1>,
           dealii::LinearAlgebra::SharedMPI::Vector<double>>(comm, i, j);
}
