cmake_minimum_required(VERSION 3.13)
project(opencalibration)

#============= OPTIONS ===============
if(NOT DEFINED OPENCALIBRATION_TESTING)
    option(OPENCALIBRATION_TESTING "" ON)
endif()

if(NOT DEFINED OPENCALIBRATION_APP)
    option(OPENCALIBRATION_APP "" ON)
endif()

# =====================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ======= DEPENDENCIES ================
find_package(Eigen3 3.3.7 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(Ceres 2.0 REQUIRED )
add_compile_definitions(CERES_USE_CXX11 TRUE)
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenCV REQUIRED opencv_imgcodecs)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(GDAL 2.4 REQUIRED)
include_directories(${GDAL_INCLUDE_DIRS})

find_package(RapidJSON 1.1 REQUIRED)
include_directories(${RAPIDJSON_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)
find_package(spdlog REQUIRED)
# =======================================

if(MSVC)
    set (CMAKE_CXX_FLAGS " /W4 /WX")
else()
    set (CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic -Werror")
    set(CMAKE_CXX_FLAGS_COVERAGE "--coverage -fprofile-arcs -ftest-coverage -O0 -fno-default-inline -fno-inline -fno-inline-small-functions -fno-elide-constructors -fprofile-abs-path"
                    CACHE STRING "Flags used by the C++ compiler during coverage builds" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "--coverage -ftest-coverage -lgcov"
                        CACHE STRING "Flags used for linking binaries during coverage builds" FORCE)
    set(CMAKE_CXX_FLAGS_ASAN "-fsanitize=address -g -O2 -DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_ASAN "-fsanitize=address -g")
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

include_directories(include)
add_subdirectory(src)

message("-- OPENCALIBRATION_TESTING: ${OPENCALIBRATION_TESTING}")
message("-- OPENCALIBRATION_APP: ${OPENCALIBRATION_APP}")

if (OPENCALIBRATION_APP)
    add_subdirectory(app)
endif()

if (OPENCALIBRATION_TESTING)
    add_subdirectory(test)
endif()
