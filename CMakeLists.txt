cmake_minimum_required(VERSION 2.6)

Project(KMeansClustering)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

include_directories(/media/portable/src/Eigen)

# Submodules
add_subdirectory(EigenHelpers)
add_subdirectory(Helpers)

add_library(libKMeansClustering KMeansClustering.cpp)
TARGET_LINK_LIBRARIES(libKMeansClustering libEigenHelpers)

ADD_EXECUTABLE(KMeansClustering Example.cpp)
TARGET_LINK_LIBRARIES(KMeansClustering libKMeansClustering libEigenHelpers)
