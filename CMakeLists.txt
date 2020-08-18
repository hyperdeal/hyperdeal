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

MESSAGE("_________________________________________________________________________________")
MESSAGE("                                                                                 ")
MESSAGE("                                 hyper.deal                                      ")
MESSAGE("_________________________________________________________________________________")


# this is the standard deal.II search mechanism

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 9.1 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

SET( TARGET_SRC hyperdeal.cc )

# Set the include directory and the name of the project
INCLUDE_DIRECTORIES(include ${CMAKE_CURRENT_BINARY_DIR}/include)

PROJECT(hyperdeal)

ADD_LIBRARY(hyperdeal  ${TARGET_SRC})

# Define custom targets to easily switch the build type:
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

DEAL_II_QUERY_GIT_INFORMATION(HYPER_DEAL)
MESSAGE("--")

MESSAGE("-- hyper.deal-version ")
MESSAGE("-- hyper.deal-branch: " ${HYPER_DEAL_GIT_BRANCH})
MESSAGE("-- hyper.deal-hash:   " ${HYPER_DEAL_GIT_REVISION})
MESSAGE("--")

MESSAGE("-- deal.II-version    ")
MESSAGE("-- deal.II-branch:    " ${DEAL_II_GIT_BRANCH})
MESSAGE("-- deal.II-hash:      " ${DEAL_II_GIT_REVISION})
MESSAGE("--")

cmake_policy(SET CMP0053 OLD)
SET(HYPER_DEAL_GIT_BRANCH "@HYPER_DEAL_GIT_BRANCH@")
SET(HYPER_DEAL_GIT_REVISION "@HYPER_DEAL_GIT_REVISION@")

CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hyper.deal/base/revision.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/hyper.deal/base/revision.h
  )

DEAL_II_SETUP_TARGET(hyperdeal)

# load macros and scripts
ADD_SUBDIRECTORY(cmake)

# build source files
ADD_SUBDIRECTORY(source)

# target the source files
TARGET_LINK_LIBRARIES(hyperdeal obj_base obj_grid)

ADD_SUBDIRECTORY(examples)


IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  SET(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)
ENDIF()

# documentation
ADD_SUBDIRECTORY(doc/doxygen)

# performance tests
ADD_SUBDIRECTORY(performance)

# tests
enable_testing()
ADD_SUBDIRECTORY(tests)
ADD_SUBDIRECTORY(examples/advection/tests)
ADD_SUBDIRECTORY(examples/vlasov_poisson/tests)