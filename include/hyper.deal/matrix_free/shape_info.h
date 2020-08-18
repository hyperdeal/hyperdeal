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
        template <int dim_x, int dim_v>
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

        /**
         * TODO
         */
        std::vector<std::vector<unsigned int>> face_orientations;
      };

      template <typename Number>
      template <int dim_x, int dim_v>
      void
      ShapeInfo<Number>::reinit(const unsigned int degree)
      {
        const unsigned int dim    = dim_x + dim_v;
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

        // clang-format off
        if (dim_x == 3 || dim_v == 3)
          {
            const unsigned int n = degree + 1;

            face_orientations.resize( 16, std::vector<unsigned int>(this->dofs_per_face));

            // x-space face
            if (dim_x == 3)
              for (unsigned int i = 0, c = 0; i < dealii::Utilities::pow(points, dim_v); ++i)
                for (unsigned int j = 0; j < n; ++j)
                  for (unsigned int k = 0; k < n; ++k, ++c)
                    {
                      // face_orientation=true,  face_flip=false, face_rotation=false
                      face_orientations[0][c] = c;
                      // face_orientation=false, face_flip=false, face_rotation=false
                      face_orientations[1][c] = j           +           k * n + i * dealii::Utilities::pow(points, 2);
                      // face_orientation=true,  face_flip=true, face_rotation=false
                      face_orientations[2][c] = (n - 1 - k) + (n - 1 - j) * n + i * dealii::Utilities::pow(points, 2);
                      // face_orientation=false, face_flip=true, face_rotation=false
                      face_orientations[3][c] = (n - 1 - j) + (n - 1 - k) * n + i * dealii::Utilities::pow(points, 2);
                      // face_orientation=true,  face_flip=false, face_rotation=true
                      face_orientations[4][c] =           j + (n - 1 - k) * n + i * dealii::Utilities::pow(points, 2);
                      // face_orientation=false, face_flip=false, face_rotation=true
                      face_orientations[5][c] =           k + (n - 1 - j) * n + i * dealii::Utilities::pow(points, 2);
                      // face_orientation=true,  face_flip=true, face_rotation=true
                      face_orientations[6][c] = (n - 1 - j) +           k * n + i * dealii::Utilities::pow(points, 2);
                      // face_orientation=false, face_flip=true, face_rotation=true
                      face_orientations[7][c] = (n - 1 - k) +           j * n + i * dealii::Utilities::pow(points, 2);
                    }
            else
              for (unsigned int c = 0; c < dealii::Utilities::pow(points, dim - 1); ++c)
                for (unsigned int i = 0; i < 8; ++i)
                  face_orientations[i][c] = c;

            // v-space face
            if (dim_v == 3)
              for (unsigned int j = 0, c = 0; j < n; ++j)
                for (unsigned int k = 0; k < n; ++k)
                  for (unsigned int i = 0; i < dealii::Utilities::pow(points, dim_x); ++i, ++c)
                    {
                      // face_orientation=true,  face_flip=false, face_rotation=false
                      face_orientations[8][c] = c;
                      // face_orientation=false, face_flip=false, face_rotation=false
                      face_orientations[9][c] = (j            +           k * n) * dealii::Utilities::pow(points, dim_x) + i;
                      // face_orientation=true,  face_flip=true, face_rotation=false
                      face_orientations[10][c] = ((n - 1 - k) + (n - 1 - j) * n) * dealii::Utilities::pow(points, dim_x) + i;
                      // face_orientation=false, face_flip=true, face_rotation=false
                      face_orientations[11][c] = ((n - 1 - j) + (n - 1 - k) * n) * dealii::Utilities::pow(points, dim_x) + i;
                      // face_orientation=true,  face_flip=false, face_rotation=true
                      face_orientations[12][c] = (j           + (n - 1 - k) * n) * dealii::Utilities::pow(points, dim_x) + i;
                      // face_orientation=false, face_flip=false, face_rotation=true
                      face_orientations[13][c] = (k           + (n - 1 - j) * n) * dealii::Utilities::pow(points, dim_x) + i;
                      // face_orientation=true,  face_flip=true, face_rotation=true
                      face_orientations[14][c] = ((n - 1 - j) +           k * n) * dealii::Utilities::pow(points, dim_x) + i;
                      // face_orientation=false, face_flip=true, face_rotation=true
                      face_orientations[15][c] = ((n - 1 - k) +           j * n) * dealii::Utilities::pow(points, dim_x) + i;
                    }
            else
              for (unsigned int c = 0; c < dealii::Utilities::pow(points, dim - 1);  ++c)
                for (unsigned int i = 8; i < 16; ++i)
                  face_orientations[i][c] = c;
          }
        else
          {
            face_to_cell_index_nodal.resize(16, std::vector<unsigned int>(1));
          }
        // clang-format on
      }


    } // namespace MatrixFreeFunctions
  }   // namespace internal
} // namespace hyperdeal

#endif