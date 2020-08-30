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

# based on https://vicrucann.github.io/tutorials/quick-cmake-doxygen/

option(HYPER_DEAL_DOCUMENTATION "Build Doxygen documentation" ON)

if (HYPER_DEAL_DOCUMENTATION)
  # check if Doxygen is installed
  find_package(Doxygen)
  if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    # copy input to output file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Configured to build Doxygen documentation")
    
    # create documentation
    add_custom_target( doxygen
      COMMAND sed -i "s|CMAKE_CURRENT_SOURCE_DIR|${CMAKE_CURRENT_SOURCE_DIR}|g" ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT} > warnings.out 2>&1
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating Doxygen documentation"
      VERBATIM )
  else (DOXYGEN_FOUND)
    message("Doxygen is not installe")
  endif (DOXYGEN_FOUND)
endif (HYPER_DEAL_DOCUMENTATION)
