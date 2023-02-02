###############################################################################
#
#  Copyright (c) 2023, Lawrence Livermore National Security, LLC
#  and other scaleUP SI developers. See the top-level COPYRIGHT
#  file for details.
#
#  SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
###############################################################################

# Require CMake version 3.12 or later to expose modern features of
# CMake for maintainability. With additional work, it is possible to
# rewrite the build system generator for CMake 3.10+, 3.8+, 3.1+,
# 3.0+, or 2.8+, with increasing amounts of work as the minimum
# required version is decreased. Notably, CMake's find modules export
# targets (e.g., FindZLIB.cmake exports the ZLIB::ZLIB IMPORTED
# target) as of version 3.1, and CMake supports using the
# <PackageName>_ROOT variables to set the prefix search path for
# find_package(<PackageName>) as of version 3.12.
cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0074 NEW) # Use <PackageName>_ROOT variables in find_package

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_Fortran_COMPILER mpif90)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Even though this project is a pure C++ library, the C language must be enabled
# for MPI and HDF5 dependency checking. The Fortran language must be enabled
# for Fortran-C name mangling convention detection.
project(scaleupflow
  VERSION 1.0.0
  DESCRIPTION "Model reduction application to scaleUP flow simulations."
  HOMEPAGE_URL "https://github.com/dreamer2368/scaleupflow"
  LANGUAGES C CXX Fortran)

## Set a bunch of variables to generate a configure header
# Enable assertion checking if debug symbols generated
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(DEBUG_CHECK_ASSERTIONS "1")
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

# Define variables for use in generating a configure file
if(GTEST_FOUND)
  set(CAROM_HAS_GTEST 1)
endif(GTEST_FOUND)

if(BLAS_FOUND)
  set(CAROM_HAVE_BLAS 1)
endif(BLAS_FOUND)

if(LAPACK_FOUND)
  set(CAROM_HAVE_LAPACK 1)
endif(LAPACK_FOUND)

if(HDF5_FOUND)
  set(CAROM_HAVE_HDF5 1)
endif(HDF5_FOUND)

# List minimum version requirements for dependencies where possible to make
# packaging easier later.
find_package(HDF5 1.8.0 REQUIRED)

find_package(BLAS 3.4.0 REQUIRED)
find_package(LAPACK 3.4.0 REQUIRED)

# In FindMPI.cmake, "CXX" refers to "the MPI C API [being] usable from C++"
find_package(MPI 1.2 REQUIRED)

find_package(ZLIB 1.2.3 REQUIRED)

find_package(Doxygen 1.8.5)

find_package(GTest 1.6.0)

# MFEM is required.
find_library(MFEM mfem "$ENV{MFEM_DIR}/lib")
find_library(HYPRE HYPRE "$ENV{HYPRE_DIR}/lib")
find_library(PARMETIS parmetis "$ENV{PARMETIS_DIR}/lib")
find_library(METIS metis "$ENV{METIS_DIR}/lib")
find_path(MFEM_INCLUDES mfem.hpp "$ENV{MFEM_DIR}/include")
find_path(HYPRE_INCLUDES HYPRE.h "$ENV{HYPRE_DIR}/include")
find_path(PARMETIS_INCLUDES metis.h "$ENV{PARMETIS_DIR}/metis/include")

include_directories(
  include
  ${MFEM_INCLUDES}
  ${HYPRE_INCLUDES}
  ${PARMETIS_INCLUDES}
  ${HDF5_C_INCLUDE_DIRS}
  ${MPI_C_INCLUDE_DIRS}
  ${MFEM_C_INCLUDE_DIRS}
)
link_libraries(
  ${MPI_C_LINK_FLAGS}
  ${MPI_C_LIBRARIES}
  MPI::MPI_C
  ${MPI_FORTRAN_LINK_FLAGS}
  ${MPI_FORTRAN_LIBRARIES}
  MPI::MPI_Fortran
  ${HDF5_LIBRARIES}
  ${LAPACK_LIBRARIES}
  ${BLAS_LIBRARIES}
  ${MFEM}
  ${HYPRE}
  ${PARMETIS}
  ${METIS}
)

set(scaleupflowObj_SOURCES

  include/navier_solver.hpp
  src/navier_solver.cpp

)

add_library(scaleupflowObj OBJECT ${scaleupflowObj_SOURCES})
# target_link_libraries(scaleupflowObj
#   PUBLIC ${MPI_C_LINK_FLAGS} ${MPI_C_LIBRARIES} MPI::MPI_C ${MPI_FORTRAN_LINK_FLAGS} ${MPI_FORTRAN_LIBRARIES} MPI::MPI_Fortran ${HDF5_LIBRARIES}
#          ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${MFEM} ${HYPRE} ${PARMETIS} ${METIS})
# target_include_directories(scaleupflowObj
#   PUBLIC ${MFEM_INCLUDES} ${HYPRE_INCLUDES} ${PARMETIS_INCLUDES} ${HDF5_C_INCLUDE_DIRS} ${MPI_C_INCLUDE_DIRS} ${MFEM_C_INCLUDE_DIRS})

add_subdirectory(bin)

## Use the C++11 standard as an entire feature instead of
## enumerating individual compiler features for simplicity
#target_compile_features(ROM PRIVATE cxx_std_11)
