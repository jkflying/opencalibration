cmake_minimum_required(VERSION 3.16)
project(opencalibration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3.7 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR}/eigen3/)

find_package(Ceres 1.14 REQUIRED )
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenCV 4.2 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(GDAL 2.4 REQUIRED)
include_directories(${GDAL_INCLUDE_DIRS})

include_directories(include)

add_subdirectory(src)
add_subdirectory(app)

add_subdirectory(test)
