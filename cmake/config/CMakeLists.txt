MESSAGE("Building CMake configuration files...")

configure_file(${CMAKE_SOURCE_DIR}/cmake/config/hyperdealConfigVersion.cmake.in
               ${CMAKE_BINARY_DIR}/hyperdealConfigVersion.cmake
               @ONLY)
configure_file(${CMAKE_SOURCE_DIR}/cmake/config/hyperdealConfig.cmake.in
               ${CMAKE_BINARY_DIR}/hyperdealConfig.cmake
               @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/hyperdealConfig.cmake ${CMAKE_BINARY_DIR}/hyperdealConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hyperdeal)
