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

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

SET(_src
  function.cc
  mpi.cc
  time_integrators.cc 
  time_loop.cc 
  utilities.cc 
  )

SET(_inst
)

ADD_LIBRARY(obj_base  OBJECT ${_src} ${_inst})

EXPAND_INSTANTIATIONS(obj_base "${_inst}")

DEAL_II_SETUP_TARGET(obj_base)
