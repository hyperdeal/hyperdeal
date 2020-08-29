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

FUNCTION(get_mpi_count _filename)

    # read file
    FILE(READ ${_filename} file_content)

    # get partitions in x-space
    STRING(REGEX MATCH "\"PartitionX\": \"([1-3])\"," _ ${file_content})
    set(_mpi_count_x ${CMAKE_MATCH_1})

    # get partitions in v-space
    STRING(REGEX MATCH "\"PartitionV\": \"([1-3])\"," _ ${file_content})
    set(_mpi_count_v ${CMAKE_MATCH_1})

    # compute overall partitions
    MATH(EXPR _mpi_count "${_mpi_count_x} * ${_mpi_count_v}")

    # return number of partitions
    SET(_mpi_count "${_mpi_count}" PARENT_SCOPE)

ENDFUNCTION(get_mpi_count)



FIND_PROGRAM(DIFF_EXECUTABLE
  NAMES diff
  HINTS ${DIFF_DIR}
  PATH_SUFFIXES bin
  )

FIND_PROGRAM(NUMDIFF_EXECUTABLE
  NAMES numdiff
  HINTS ${NUMDIFF_DIR}
  PATH_SUFFIXES bin
  )


MARK_AS_ADVANCED(DIFF_EXECUTABLE NUMDIFF_EXECUTABLE)

IF("${TEST_DIFF}" STREQUAL "")
  IF(NOT NUMDIFF_EXECUTABLE MATCHES "-NOTFOUND")
    SET(TEST_DIFF ${NUMDIFF_EXECUTABLE} -a 1e-5 -r 1e-8 -s ' \\t\\n:,')
    IF(DIFF_EXECUTABLE MATCHES "-NOTFOUND")
      SET(DIFF_EXECUTABLE ${NUMDIFF_EXECUTABLE})
    ENDIF()
  ELSEIF(NOT DIFF_EXECUTABLE MATCHES "-NOTFOUND")
    SET(TEST_DIFF ${DIFF_EXECUTABLE})
  ELSE()
    MESSAGE(FATAL_ERROR
      "Could not find diff or numdiff. One of those are required for running the tests.\n"
      "Please specify TEST_DIFF by hand."
      )
  ENDIF()
ENDIF()

ADD_CUSTOM_TARGET(tests_vp)



FILE(GLOB _testexes *.configuration)
LIST(SORT _testexes)

SET(_n_exes "0")
SET(_n_tests "0")
FOREACH(_testexe ${_testexes})

   file(READ ${_testexe} FLAGS)

    GET_FILENAME_COMPONENT(_testexe ${_testexe} NAME_WE)
    MATH(EXPR _n_exes "${_n_exes} + 1")  



    ADD_EXECUTABLE( ${_testexe} ../vlasov_poisson.cc)

    separate_arguments(FLAGS)
    target_compile_definitions(${_testexe} PRIVATE ${FLAGS})

    DEAL_II_SETUP_TARGET(${_testexe})
    TARGET_LINK_LIBRARIES(${_testexe} "hyperdeal")



  FILE(GLOB _tests ${_testexe}.*.json)
    LIST (SORT _tests)
    FOREACH(_test ${_tests})

        SET(TEST_ABS ${_test})

        GET_MPI_COUNT(${_test})

        GET_FILENAME_COMPONENT(_test ${_test} NAME_WLE)

        MATH(EXPR _n_tests "${_n_tests} + 1")

        FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output-${_test})

        ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/time_history_diagnostic.out
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}
          COMMAND rm -f * 
          COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.out time_history_diagnostic.out.ref
          COMMAND echo pwd
          COMMAND pwd
          COMMAND mpirun -np ${_mpi_count} --oversubscribe ../${_testexe} ${TEST_ABS} > screen-output
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.json ${_testdepends} ${_testexe}
          )

        ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/time_history_diagnostic.out.ref
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}
          COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.out time_history_diagnostic.out.ref
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/time_history_diagnostic.out  ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.out
        )

        # The final target for this test
        ADD_CUSTOM_TARGET(${_test}.run
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/time_history_diagnostic.out ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/time_history_diagnostic.out.ref
          )

        ADD_CUSTOM_TARGET(tests_vp.${_test})

        # create the target that compares the .notime with the saved file
        ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/diff
          COMMAND
              if (${TEST_DIFF} ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/time_history_diagnostic.out.ref
                    ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/time_history_diagnostic.out
                  > ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff) \; then
                : \;
              else
                mv ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff
                   ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff.failed \;
                echo "${_test}: BUILD successful." \;
                echo "${_test}: RUN successful." \;
                echo "${_test}: DIFF failed. ------ Source: ${CMAKE_CURRENT_SOURCE_DIR}/${_test}.out" \;
                echo "${_test}: DIFF failed. ------ Result: ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/time_history_diagnostic.out" \;
                echo "${_test}: DIFF failed. ------ Diff:   ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff.failed" \;
                echo "${_test}: DIFF failed. ------ First 20 lines of numdiff output:" \;
                head -n 20 ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/${_test}.diff.failed \;
                false \;
              fi
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/time_history_diagnostic.out ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/time_history_diagnostic.out.ref
        )

        ADD_CUSTOM_TARGET(${_test}.diff
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/output-${_test}/diff
          )

        ADD_DEPENDENCIES(tests_vp ${_test}.diff)
        ADD_TEST(NAME ${_test}
          COMMAND
          ${CMAKE_COMMAND}
          -DBINARY_DIR=${CMAKE_BINARY_DIR}
          -DTESTNAME=${_test}
          -DERROR="Test ${_test} failed"
          -P ${CMAKE_SOURCE_DIR}/examples/vlasov_poisson/tests/run_test.cmake
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )


    ENDFOREACH()


ENDFOREACH()

MESSAGE(STATUS "Added ${_n_exes} vlasov_poisson test executables")
MESSAGE(STATUS "Added ${_n_tests} vlasov_poisson tests")
