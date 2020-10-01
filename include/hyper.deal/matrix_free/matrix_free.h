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

#ifndef HYPERDEAL_NDIM_MATRIXFREE
#define HYPERDEAL_NDIM_MATRIXFREE

#include <hyper.deal/base/config.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <hyper.deal/base/memory_consumption.h>
#include <hyper.deal/base/timers.h>
#include <hyper.deal/lac/sm_vector.h>
#include <hyper.deal/matrix_free/dof_info.h>
#include <hyper.deal/matrix_free/face_info.h>
#include <hyper.deal/matrix_free/id.h>
#include <hyper.deal/matrix_free/shape_info.h>
#include <hyper.deal/matrix_free/vector_partitioner.h>

namespace hyperdeal
{
  /**
   * A matrix-free class for phase-space.
   */
  template <int dim_x, int dim_v, typename Number, typename VectorizedArrayType>
  class MatrixFree
  {
  public:
    using ID = TensorID;

    using VectorizedArrayTypeX = VectorizedArrayType;
    using VectorizedArrayTypeV = dealii::VectorizedArray<Number, 1>;

    /**
     * Type of access on faces.
     *
     * @note Currently only none and values (ghost-face access) are supported.
     */
    enum class DataAccessOnFaces
    {
      none,
      values,
      gradients,
      unspecified
    };

    /**
     * Struct to configure MatrixFree.
     */
    struct AdditionalData
    {
      /**
       * Constructor.
       */
      AdditionalData()
        : do_ghost_faces(true)
        , do_buffering(false)
        , use_ecl(true)
        , overlapping_level(0 /*no overlapping communication-computation*/)
      {}

      /**
       * Work on ghost faces or ghost cells.
       *
       * @note Currently only ghost faces are supported.
       */
      bool do_ghost_faces;

      /**
       * Do buffering of values on shared memory domain.
       *
       * TODO: extend partitioner so that it can work in buffering and
       *   non-buffering at the same time (maybe connected to ECL).
       */
      bool do_buffering;

      /**
       * Use element centric loops.
       *
       * TODO: extend partitioner so that it can work for ECL and FCL the same
       *   time.
       */
      bool use_ecl;

      /**
       * TODO
       */
      unsigned int overlapping_level;
    };


    /**
     * Constructor (does nothing - see reinit()).
     */
    MatrixFree(const MPI_Comm comm,
               const MPI_Comm comm_sm,
               const dealii::MatrixFree<dim_x, Number, VectorizedArrayTypeX>
                 &matrix_free_x,
               const dealii::MatrixFree<dim_v, Number, VectorizedArrayTypeV>
                 &matrix_free_v);

    /**
     * Actually setup internal data structures (except partitioner).
     */
    void
    reinit(const AdditionalData &ad = AdditionalData());

    /**
     * Loop over all cell pairs. No communication performed.
     */
    template <typename OutVector, typename InVector>
    void
    cell_loop(
      const std::function<
        void(const MatrixFree &, OutVector &, const InVector &, const ID)>
        &             cell_operation,
      OutVector &     dst,
      const InVector &src) const;

    /**
     * The same as above, but without std::function.
     */
    template <typename CLASS, typename OutVector, typename InVector>
    void
    cell_loop(void (CLASS::*cell_operation)(const MatrixFree &,
                                            OutVector &,
                                            const InVector &,
                                            const ID),
              CLASS *         owning_class,
              OutVector &     dst,
              const InVector &src) const;

    /**
     * Loop over all cell pairs in an element-centric fashion (ECL). It
     * includes a ghost value update of the source vector.
     */
    template <typename OutVector, typename InVector>
    void
    loop_cell_centric(
      const std::function<
        void(const MatrixFree &, OutVector &, const InVector &, const ID)>
        &                     cell_operation,
      OutVector &             dst,
      const InVector &        src,
      const DataAccessOnFaces src_vector_face_access =
        DataAccessOnFaces::unspecified,
      Timers *timers = nullptr) const;

    /**
     * The same as above, but without std::function.
     */
    template <typename CLASS, typename OutVector, typename InVector>
    void
    loop_cell_centric(void (CLASS::*cell_operation)(const MatrixFree &,
                                                    OutVector &,
                                                    const InVector &,
                                                    const ID),
                      CLASS *                 owning_class,
                      OutVector &             dst,
                      const InVector &        src,
                      const DataAccessOnFaces src_vector_face_access =
                        DataAccessOnFaces::unspecified,
                      Timers *timers = nullptr) const;

    /**
     * Loop over all cell pairs in an face-centric fashion (FCL). It
     * includes a ghost value update of the source vector and a compression
     * of the destination vector.
     */
    template <typename OutVector, typename InVector>
    void
    loop(const std::function<
           void(const MatrixFree &, OutVector &, const InVector &, const ID)>
           &cell_operation,
         const std::function<
           void(const MatrixFree &, OutVector &, const InVector &, const ID)>
           &face_operation,
         const std::function<
           void(const MatrixFree &, OutVector &, const InVector &, const ID)>
           &                     boundary_operation,
         OutVector &             dst,
         const InVector &        src,
         const DataAccessOnFaces dst_vector_face_access =
           DataAccessOnFaces::unspecified,
         const DataAccessOnFaces src_vector_face_access =
           DataAccessOnFaces::unspecified,
         Timers *timers = nullptr) const;

    /**
     * Loop over all cell pairs in an face-centric fashion (FCL). It
     * includes a ghost value update of the source vector and a compression
     * of the destination vector.
     */
    template <typename CLASS, typename OutVector, typename InVector>
    void
    loop(void (CLASS::*cell_operation)(const MatrixFree &,
                                       OutVector &,
                                       const InVector &,
                                       const ID),
         void (CLASS::*face_operation)(const MatrixFree &,
                                       OutVector &,
                                       const InVector &,
                                       const ID),
         void (CLASS::*boundary_operation)(const MatrixFree &,
                                           OutVector &,
                                           const InVector &,
                                           const ID),
         CLASS *                 owning_class,
         OutVector &             dst,
         const InVector &        src,
         const DataAccessOnFaces dst_vector_face_access =
           DataAccessOnFaces::unspecified,
         const DataAccessOnFaces src_vector_face_access =
           DataAccessOnFaces::unspecified,
         Timers *timers = nullptr) const;

    /**
     * Return global communicator.
     */
    const MPI_Comm &
    get_communicator() const;

    /**
     * Allocate shared-memory vector. Additionally, all values are zeroed out
     * so that out-of-memory errors are back-trackable.
     *
     * @note If this function has been called the first time, also the
     *   partitioner is set up.
     */
    void
    initialize_dof_vector(dealii::LinearAlgebra::SharedMPI::Vector<Number> &vec,
                          const unsigned int dof_handler_index = 0,
                          const bool         do_ghosts         = true,
                          const bool         zero_out_values   = true) const;

    /**
     * Return boundary id of face.
     */
    dealii::types::boundary_id
    get_boundary_id(const ID macro_face) const;

    /**
     * Return boundary id of face of cell.
     *
     * @note In constrast to dealii::MatrixFree::get_faces_by_cells_boundary_id()
     *   we only return a single number since we require that the cells
     *   of a macro cell all have the same type of boundary faces.
     */
    dealii::types::boundary_id
    get_faces_by_cells_boundary_id(const TensorID &   macro_cell,
                                   const unsigned int face_number) const;

    /**
     * Is ECL supported?
     */
    bool
    is_ecl_supported() const;

    /**
     * Return if ghost faces or ghost cells are supported.
     */
    bool
    are_ghost_faces_supported() const;

    /**
     * Return dealii::MatrixFree for x-space.
     */
    const dealii::MatrixFree<dim_x, Number, VectorizedArrayTypeX> &
    get_matrix_free_x() const;

    /**
     * Return dealii::MatrixFree for v-space.
     */
    const dealii::MatrixFree<dim_v, Number, VectorizedArrayTypeV> &
    get_matrix_free_v() const;

    /**
     * Return dof info.
     */
    const internal::MatrixFreeFunctions::DoFInfo &
    get_dof_info() const;

    /**
     * Return face info.
     */
    const internal::MatrixFreeFunctions::FaceInfo &
    get_face_info() const;

    /**
     * Return shape info.
     */
    const internal::MatrixFreeFunctions::ShapeInfo<Number> &
    get_shape_info() const;

    /**
     * Return an estimate for the memory consumption, in bytes, of this object.
     */
    MemoryConsumption
    memory_consumption() const;

    /**
     * Return partitioner of the vectors.
     */
    const std::shared_ptr<
      const dealii::LinearAlgebra::SharedMPI::PartitionerBase> &
    get_vector_partitioner() const;


  private:
    /**
     * Global communicator.
     */
    const MPI_Comm comm;

    /**
     * Shared-memory communicator.
     */
    const MPI_Comm comm_sm;

    /**
     * dealii::MatrixFree for x-space.
     */
    const dealii::MatrixFree<dim_x, Number, VectorizedArrayTypeX>
      &matrix_free_x;

    /**
     * dealii::MatrixFree for v-space.
     */
    const dealii::MatrixFree<dim_v, Number, VectorizedArrayTypeV>
      &matrix_free_v;

    /**
     * Configuration: buffer shared-memory ghost values.
     */
    bool do_buffering;

    /**
     * Configuration: work with ghost faces (advection) or ghost cell (Laplace).
     */
    bool do_ghost_faces;

    /**
     * Perform ECL or FCL.
     */
    bool use_ecl;

    /**
     * Partitioner for ghost_value_update() and compress().
     */
    std::shared_ptr<const dealii::LinearAlgebra::SharedMPI::PartitionerBase>
      partitioner;

    /**
     * Cell info.
     */
    internal::MatrixFreeFunctions::DoFInfo dof_info;

    /**
     * Face info.
     */
    internal::MatrixFreeFunctions::FaceInfo face_info;

    /**
     * Shape info.
     */
    internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info;

    /**
     * Partitions for ECL.
     *
     * TODO: more details
     */
    std::array<std::vector<ID>, 3> partitions;
  };

} // namespace hyperdeal

#include <hyper.deal/matrix_free/matrix_free.templates.h>

#endif
