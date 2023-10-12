CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 QUIET
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
PROJECT("exmples")

DEAL_II_INITIALIZE_CACHED_VARIABLES()

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
  )

file(GLOB files "*.cc")
foreach(file ${files})
GET_FILENAME_COMPONENT(file_name ${file} NAME)

  STRING( FIND ${file_name} ".likwid" LIKWID_TO_BE_USED)

  IF(${LIKWID_TO_BE_USED})
    STRING( REPLACE ".likwid" "" file_name ${file_name} )
  ENDIF()  

  STRING( REPLACE ".cc" "" app ${file_name} )

  ADD_EXECUTABLE(${app} ${file})
  TARGET_COMPILE_DEFINITIONS(${app} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  DEAL_II_SETUP_TARGET(${app})

  target_include_directories(${app} PUBLIC "include/")

  IF(${LIKWID_TO_BE_USED})
    FIND_LIBRARY(LIKWID likwid HINTS $ENV{LIKWID_LIB})
    
    IF(LIKWID)
      TARGET_COMPILE_DEFINITIONS(${app} PUBLIC LIKWID_PERFMON)
      TARGET_INCLUDE_DIRECTORIES(${app} PUBLIC $ENV{LIKWID_INCLUDE})
      TARGET_LINK_LIBRARIES(${app} ${LIKWID})
    ENDIF()
  ENDIF()

endforeach()

IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  SET(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)
ENDIF()

enable_testing()

DEAL_II_PICKUP_TESTS("tests")
