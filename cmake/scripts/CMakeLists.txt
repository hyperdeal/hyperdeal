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

# source: https://github.com/dealii/dealii/blob/master/cmake/macros/macro_expand_instantiations.cmake
# date:   16.08.2020 
MACRO(EXPAND_INSTANTIATIONS _target _inst_in_files)

  FOREACH (_inst_in_file ${_inst_in_files})
    STRING(REGEX REPLACE "\\.in$" "" _inst_file "${_inst_in_file}" )

    IF(NOT CMAKE_CROSSCOMPILING)
      SET(_command expand_instantiations_exe)
      SET(_dependency expand_instantiations_exe)
    ELSE()
      SET(_command expand_instantiations)
      SET(_dependency)
    ENDIF()

    # create a .inst.tmp file first and only move to the correct name if the
    # first call succeeds. Otherwise we might be generating an incomplete
    # .inst file
    ADD_CUSTOM_COMMAND(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_inst_file}
      DEPENDS ${_dependency}
              ${CMAKE_SOURCE_DIR}/contrib/template-arguments
              ${CMAKE_CURRENT_SOURCE_DIR}/${_inst_in_file}
      COMMAND ${_command}
      ARGS ${CMAKE_SOURCE_DIR}/contrib/template-arguments
           < ${CMAKE_CURRENT_SOURCE_DIR}/${_inst_in_file}
           > ${CMAKE_CURRENT_BINARY_DIR}/${_inst_file}.tmp
      COMMAND ${CMAKE_COMMAND}
      ARGS -E rename
           ${CMAKE_CURRENT_BINARY_DIR}/${_inst_file}.tmp
           ${CMAKE_CURRENT_BINARY_DIR}/${_inst_file}
      )

    LIST(APPEND _inst_targets ${CMAKE_CURRENT_BINARY_DIR}/${_inst_file})
  ENDFOREACH()

  #
  # Define a custom target that depends on the generation of all inst.in
  # files.
  #
  ADD_CUSTOM_TARGET(${_target}_inst ALL DEPENDS ${_inst_targets})

  #
  # Provide a way to generate all .inst files with a custom target.
  #
  ADD_DEPENDENCIES(expand_all_instantiations ${_target}_inst)

  #
  # Add a dependency to all target.${build_type} so that target.inst is
  # fully generated before target will be processed.
  #
  ADD_DEPENDENCIES(${_target} ${_target}_inst)

ENDMACRO()

ADD_CUSTOM_TARGET(expand_all_instantiations)

ADD_EXECUTABLE(expand_instantiations_exe expand_instantiations.cc)
DEAL_II_SETUP_TARGET(expand_instantiations_exe)
