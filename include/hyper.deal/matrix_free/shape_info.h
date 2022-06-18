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

      namespace
      {
        template <int dim>
        void
        fill_face_to_cell_index_nodal(
          const unsigned int              points,
          dealii::Table<2, unsigned int> &face_to_cell_index_nodal)
        {
          // adopted from dealii::internal::ShapeInfo::reinit()

#ifdef DEBUG
          const unsigned int dofs_per_component_on_cell =
            dealii::Utilities::pow(points, dim);
#endif
          const unsigned int dofs_per_component_on_face =
            dealii::Utilities::pow(points, dim - 1);
          face_to_cell_index_nodal.reinit(
            dealii::GeometryInfo<dim>::faces_per_cell,
            dofs_per_component_on_face);
          for (const auto f : dealii::GeometryInfo<dim>::face_indices())
            {
              const unsigned int direction = f / 2;
              const unsigned int stride    = direction < dim - 1 ? points : 1;
              int                shift     = 1;
              for (unsigned int d = 0; d < direction; ++d)
                shift *= points;
              const unsigned int offset = (f % 2) * (points - 1) * shift;

              if (direction == 0 || direction == dim - 1)
                for (unsigned int i = 0; i < dofs_per_component_on_face; ++i)
                  face_to_cell_index_nodal(f, i) = offset + i * stride;
              else
                // local coordinate system on faces 2 and 3 is zx in
                // deal.II, not xz as expected for tensor products -> adjust
                // that here
                for (unsigned int j = 0; j < points; ++j)
                  for (unsigned int i = 0; i < points; ++i)
                    {
                      const unsigned int ind =
                        offset + j * dofs_per_component_on_face + i;
                      AssertIndexRange(ind, dofs_per_component_on_cell);
                      const unsigned int l           = i * points + j;
                      face_to_cell_index_nodal(f, l) = ind;
                    }
            }
        }
      } // namespace

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

        dealii::Table<2, unsigned int> face_to_cell_index_nodal_x;
        dealii::Table<2, unsigned int> face_to_cell_index_nodal_v;

        fill_face_to_cell_index_nodal<dim_x>(points,
                                             face_to_cell_index_nodal_x);
        fill_face_to_cell_index_nodal<dim_v>(points,
                                             face_to_cell_index_nodal_v);

        for (unsigned int surface = 0;
             surface < dealii::GeometryInfo<dim>::faces_per_cell;
             surface++)
          {
            face_to_cell_index_nodal[surface].resize(
              dofs_per_component_on_face);

            if (surface < dim_x * 2)
              for (int i = 0, k = 0; i < std::pow(points, dim_v); i++)
                for (int j = 0; j < std::pow(points, dim_x - 1); j++)
                  face_to_cell_index_nodal[surface][k++] =
                    face_to_cell_index_nodal_x(surface, j) +
                    std::pow(points, dim_x) * i;
            else
              for (int i = 0, k = 0; i < std::pow(points, dim_v - 1); i++)
                for (int j = 0; j < std::pow(points, dim_x); j++)
                  face_to_cell_index_nodal[surface][k++] =
                    j + std::pow(points, dim_x) *
                          face_to_cell_index_nodal_v(surface - 2 * dim_x, i);
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