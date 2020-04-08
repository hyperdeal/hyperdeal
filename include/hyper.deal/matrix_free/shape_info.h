#ifndef HYPERDEAL_NDIM_MATRIXFREE_SHAPE_INFO
#define HYPERDEAL_NDIM_MATRIXFREE_SHAPE_INFO

#include <hyper.deal/base/config.h>

#include <deal.II/base/geometry_info.h>

namespace hyperdeal
{
  namespace internal
  {
    namespace MatrixFreeFunctions
    {
      /**
       * Utility functions for shape functions.
       */
      template <typename Number>
      struct ShapeInfo
      {
        /**
         * Setup internal data structures for a given @p degree.
         *
         * TODO: take a dealii::FiniteElement?
         */
        template <int dim>
        void
        reinit(const unsigned int degree);

        std::size_t
        memory_consumption() const
        {
          return dealii::MemoryConsumption::memory_consumption(
            face_to_cell_index_nodal);
        }

        /**
         * Degrees of freedom per cell.
         */
        dealii::types::global_dof_index dofs_per_cell;

        /**
         * Degrees of freedom per face.
         */
        dealii::types::global_dof_index dofs_per_face;

        /**
         * Indices of degrees of freedom of the 2*dim faces.
         *
         * @note This is similar to
         *   dealii::internal::MatrixFreeFunctions::ShapeInfo without
         *   the annoying orientation switch in 3D.
         */
        std::vector<std::vector<unsigned int>> face_to_cell_index_nodal;
      };

      template <typename Number>
      template <int dim>
      void
      ShapeInfo<Number>::reinit(const unsigned int degree)
      {
        const unsigned int points = degree + 1;

        this->dofs_per_cell = dealii::Utilities::pow(points, dim);
        this->dofs_per_face = dealii::Utilities::pow(points, dim - 1);

        const unsigned int dofs_per_component_on_face =
          dealii::Utilities::pow(points, dim - 1);

        face_to_cell_index_nodal.resize(
          dealii::GeometryInfo<dim>::faces_per_cell);

        for (unsigned int surface = 0;
             surface < dealii::GeometryInfo<dim>::faces_per_cell;
             surface++)
          {
            face_to_cell_index_nodal[surface].resize(
              dofs_per_component_on_face);
            // create indices for one surfaces
            const unsigned int d = surface / 2; // direction
            const unsigned int s = surface % 2; // left or right surface

            for (int i = 0, k = 0; i < std::pow(points, dim - d - 1); i++)
              for (int j = 0; j < std::pow(points, d); j++)
                face_to_cell_index_nodal[surface][k++] =
                  i * std::pow(points, d + 1) +
                  (s == 0 ? 0 : (points - 1)) * std::pow(points, d) + j;
          }
      }


    } // namespace MatrixFreeFunctions
  }   // namespace internal
} // namespace hyperdeal

#endif