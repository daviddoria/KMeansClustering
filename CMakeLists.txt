cmake_minimum_required(VERSION 2.8)

Project(KMeansClustering)
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(CMakeHelpers)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

#### Eigen ####
# Tell CMake to also look in the source directory to find some .cmake files
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

# Set the CMake variable to the value of the environment variable.
# If the build is absolutely fresh, this will be the case if the CMake variable EIGEN3_INCLUDE_DIR is not set.
if(NOT EIGEN3_INCLUDE_DIR)
  set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
  message("Set EIGEN3_INCLUDE_DIR to ${EIGEN3_INCLUDE_DIR} from environment variable of the same name.")
else()
  message("EIGEN3_INCLUDE_DIR is ${EIGEN3_INCLUDE_DIR}")
endif()

if(NOT EIGEN3_FOUND)
  FIND_PACKAGE(Eigen3 REQUIRED) #requires FindEigen3.cmake to be in the source directory
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Submodules
UseSubmodule(Helpers KMeansClustering)

add_library(KMeansClustering KMeansClustering.cpp)

CreateSubmodule(KMeansClustering)

SET(KMeansClustering_BuildExamples OFF CACHE BOOL "KMeansClustering_BuildExamples")
if(KMeansClustering_BuildExamples)
  ADD_EXECUTABLE(KMeansClusteringExample KMeansClusteringExample.cpp)
  TARGET_LINK_LIBRARIES(KMeansClusteringExample KMeansClustering)
endif(KMeansClustering_BuildExamples)
