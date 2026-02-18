cmake_minimum_required(VERSION 3.13)
project(opencalibration)

#============= OPTIONS ===============
if(NOT DEFINED OPENCALIBRATION_TESTING)
    option(OPENCALIBRATION_TESTING "" ON)
endif()

if(NOT DEFINED OPENCALIBRATION_APP)
    option(OPENCALIBRATION_APP "" ON)
endif()

option(OPENCALIBRATION_CLANG_TIDY "Run clang-tidy during compilation" OFF)

# =====================================

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ======= DEPENDENCIES ================
find_package(Eigen3 3.3.7 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})

find_package(Ceres 2.0 REQUIRED )
add_compile_definitions(CERES_USE_CXX11 TRUE)
include_directories(SYSTEM ${CERES_INCLUDE_DIRS})

find_package(OpenCV REQUIRED HINTS external/install/lib/cmake/)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

find_package(GDAL 2.4 REQUIRED)
include_directories(SYSTEM ${GDAL_INCLUDE_DIRS})

find_package(RapidJSON 1.1 REQUIRED)
include_directories(SYSTEM ${RAPIDJSON_INCLUDE_DIRS})

include_directories(SYSTEM external/unordered_dense/include)

find_package(OpenMP REQUIRED)
find_package(spdlog REQUIRED)
# =======================================

if(MSVC)
    set (CMAKE_CXX_FLAGS " /W4 /WX")
else()
    set (CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic -Werror -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_COVERAGE "--coverage -fprofile-arcs -ftest-coverage -Og -fno-default-inline -fno-inline -fno-inline-small-functions -fno-elide-constructors -fprofile-abs-path -fprofile-update=atomic"
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

if(OPENCALIBRATION_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE
        NAMES clang-tidy-18 clang-tidy-17 clang-tidy-16 clang-tidy-15 clang-tidy-14 clang-tidy
        REQUIRED)
    message(STATUS "clang-tidy: ${CLANG_TIDY_EXE}")
    # Use clang's resource dir so clang-tidy can find omp.h (libomp-18-dev) without GCC's incompatible headers.
    string(REPLACE "clang-tidy" "clang" CLANG_EXE "${CLANG_TIDY_EXE}")
    execute_process(COMMAND ${CLANG_EXE} --print-resource-dir
                    OUTPUT_VARIABLE CLANG_RESOURCE_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE CLANG_RESOURCE_RESULT)
    if(CLANG_RESOURCE_RESULT EQUAL 0)
        message(STATUS "clang resource dir: ${CLANG_RESOURCE_DIR}")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}"
            "--extra-arg=-isystem${CLANG_RESOURCE_DIR}/include")
    else()
        message(WARNING "Could not determine clang resource dir; falling back to -U_OPENMP")
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}" "--extra-arg=-U_OPENMP")
    endif()
endif()

# Camera database bundled resource
set(CAMERA_DATABASE_SOURCE "${CMAKE_SOURCE_DIR}/data/camera_database.json")
set(CAMERA_DATABASE_DEST "${CMAKE_BINARY_DIR}/camera_database.json")
configure_file(${CAMERA_DATABASE_SOURCE} ${CAMERA_DATABASE_DEST} COPYONLY)

add_subdirectory(src)

message("-- OPENCALIBRATION_TESTING: ${OPENCALIBRATION_TESTING}")
message("-- OPENCALIBRATION_APP: ${OPENCALIBRATION_APP}")

if (OPENCALIBRATION_APP)
    add_subdirectory(app)
endif()

if (OPENCALIBRATION_TESTING)
    add_subdirectory(test)
endif()
