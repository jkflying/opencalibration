cmake_minimum_required(VERSION 3.16)
project(opencalibration)

find_package(Eigen3 3.3.7 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR}/eigen3/)

find_package(Ceres 1.14 REQUIRED )

find_package(OpenCV 4.2 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(include)

add_subdirectory(src)

add_subdirectory(test)
