## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the hyper.deal authors
##
## This file is part of the hyper.deal library.
##
## The hyper.deal library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 3.0 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.MD at
## the top level directory of hyper.deal.
##
## ---------------------------------------------------------------------

SET(TARGET_SRC vlasov_poisson.cc)

ADD_EXECUTABLE( "vlasov_poisson" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("vlasov_poisson")
TARGET_LINK_LIBRARIES("vlasov_poisson" "hyperdeal")