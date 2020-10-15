#ifndef HYPERDEAL_NDIM_MATRIXFREE_DOF_INFO
#define HYPERDEAL_NDIM_MATRIXFREE_DOF_INFO

#include <hyper.deal/base/config.h>

namespace hyperdeal
{
  namespace internal
  {
    namespace MatrixFreeFunctions
    {
      /**
       * The class that stores the indices of the degrees of freedom for all the
       * cells and faces within the shared-memory domain. As a consequence
       * the index is a pair consisting of:
       *   - the rank of the process owning the cell (with the shared memory)
       *   - the offset from the beginning of the array of that rank.
       */
      struct DoFInfo
      {
        /**
         * Caches the number of indices filled when vectorizing.
         *
         * @note (0) FCL (interior); (1) FCL (exterior);
         *       (2) cell; (3) ECL (exterior - TODO remove should be the same
         *       as (2))
         */
        std::array<std::vector<unsigned char>, 4> n_vectorization_lanes_filled;

        /**
         * Stores the indices of the degrees of freedom for each face/cell
         * in the shared-memory domain. The first number of the pair
         * is the rank within the shared-memory communicator and the
         * second number is the offset with the corresponding local array.
         *
         * @note Faces might be stand alone or might be embedded into
         *   cells: ghost faces vs. faces of cells owned by a process in
         *   the same shared memory domain. The number here gives the
         *   starting point of the corresponding primitive. If a face
         *   is embedded into a cell additional info is needed from
         *   FaceInfo for reading/writing.
         *
         * @note (0) FCL (interior); (1) FCL (exterior);
         *       (2) cell; (3) ECL (exterior)
         */
        std::array<std::vector<std::pair<unsigned int, unsigned int>>, 4>
          dof_indices_contiguous_ptr;


        std::size_t
        memory_consumption() const
        {
          return dealii::MemoryConsumption::memory_consumption(
                   n_vectorization_lanes_filled) +
                 dealii::MemoryConsumption::memory_consumption(
                   dof_indices_contiguous_ptr);
        }
      };
    } // namespace MatrixFreeFunctions
  }   // namespace internal
} // namespace hyperdeal

#endif
