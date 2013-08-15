# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ && make -j2

# ----------------------------------------------------------------------
# Base CMake setup
cmake_minimum_required (VERSION 2.6)

#---------------  set gpu files -----------------
find_package(CUDA 4.0 REQUIRED)
# collect source files
#file( GLOB  hdr *.hpp *.h )
file( GLOB  src_cuda  *.cu)

#if(not PRO_BUILD_DIR)
#MESSAGE(ERROR "PRO_BUILD_DIR is missing")
#endif(PRO_BUILD_DIR)

MESSAGE(STATUS "project dir is: " ${PROJECT_BINARY_DIR})
MESSAGE(STATUS "cuda files: " ${src_cuda})

CUDA_ADD_LIBRARY(yl_cuda_code STATIC ${src_cuda} )

#SET(LIBRARY_OUTPUT_PATH ./lib)


