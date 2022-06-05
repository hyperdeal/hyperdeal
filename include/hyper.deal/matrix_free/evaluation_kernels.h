#ifndef HYPERDEAL_NDIM_MATRIXFREE_EVALUTION_KERNELS
#define HYPERDEAL_NDIM_MATRIXFREE_EVALUTION_KERNELS

#include <hyper.deal/base/config.h>

namespace hyperdeal
{
  namespace internal
  {
    template <int dim_x, int dim_v, int fe_degree, typename Number>
    using FEFaceNormalEvaluationImpl = dealii::internal::
      FEFaceNormalEvaluationImpl<dim_x + dim_v, fe_degree, Number, true>;

  } // namespace internal
} // namespace hyperdeal

#endif
