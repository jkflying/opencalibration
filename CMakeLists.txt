cmake_minimum_required(VERSION 3.14)
project(opencalibration VERSION 0.1.0)

#============= OPTIONS ===============
if(NOT DEFINED OPENCALIBRATION_TESTING)
    option(OPENCALIBRATION_TESTING "" ON)
endif()

if(NOT DEFINED OPENCALIBRATION_APP)
    option(OPENCALIBRATION_APP "" ON)
endif()

option(OPENCALIBRATION_CLANG_TIDY "Run clang-tidy during compilation" OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# =====================================

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

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

find_package(OpenCV REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

find_package(GDAL 2.4 REQUIRED)
include_directories(SYSTEM ${GDAL_INCLUDE_DIRS})

find_package(RapidJSON 1.1 REQUIRED)
include_directories(SYSTEM ${RAPIDJSON_INCLUDE_DIRS})

include_directories(SYSTEM external/unordered_dense/include)

find_package(OpenMP REQUIRED)
find_package(spdlog REQUIRED)
find_package(mimalloc REQUIRED)
# =======================================

if(MSVC)
    set (CMAKE_CXX_FLAGS " /W4 /WX")
else()
    set (CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic -Werror -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_COVERAGE "--coverage -fprofile-arcs -ftest-coverage -Og -g -fprofile-abs-path -fprofile-update=atomic"
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
    find_program(CLANG_TIDY_CACHE_EXE cltcache
        HINTS "${CMAKE_SOURCE_DIR}/external/venv/bin")
    if(CLANG_TIDY_CACHE_EXE)
        message(STATUS "cltcache: ${CLANG_TIDY_CACHE_EXE}")
        set(CLANG_TIDY_LAUNCHER "${CLANG_TIDY_CACHE_EXE}")
    else()
        set(CLANG_TIDY_LAUNCHER "")
    endif()

    if(CLANG_RESOURCE_RESULT EQUAL 0)
        message(STATUS "clang resource dir: ${CLANG_RESOURCE_DIR}")
        set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_LAUNCHER} "${CLANG_TIDY_EXE}"
            "--extra-arg=-isystem${CLANG_RESOURCE_DIR}/include")
    else()
        message(WARNING "Could not determine clang resource dir; falling back to -U_OPENMP")
        set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_LAUNCHER} "${CLANG_TIDY_EXE}" "--extra-arg=-U_OPENMP")
    endif()
endif()

# Camera database bundled resource
set(CAMERA_DATABASE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/data/camera_database.json")
set(CAMERA_DATABASE_DEST "${CMAKE_BINARY_DIR}/camera_database.json")
configure_file(${CAMERA_DATABASE_SOURCE} ${CAMERA_DATABASE_DEST} COPYONLY)

add_subdirectory(src)

set(OPENCALIBRATION_LIBRARIES
    oc_dense oc_distort oc_extract oc_geo_coord oc_geometry
    oc_io oc_match oc_model_inliers oc_ortho oc_performance
    oc_pipeline oc_relax oc_surface oc_tile_ordering)

# Namespace aliases for add_subdirectory() consumers
foreach(_target IN LISTS OPENCALIBRATION_LIBRARIES)
    add_library(opencalibration::${_target} ALIAS ${_target})
endforeach()

message("-- OPENCALIBRATION_TESTING: ${OPENCALIBRATION_TESTING}")
message("-- OPENCALIBRATION_APP: ${OPENCALIBRATION_APP}")

if (OPENCALIBRATION_APP)
    add_subdirectory(app)
endif()

if (OPENCALIBRATION_TESTING)
    add_subdirectory(test)
endif()

# ============= INSTALL ==============
install(TARGETS ${OPENCALIBRATION_LIBRARIES}
    EXPORT opencalibrationTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/opencalibration
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY external/unordered_dense/include/ankerl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES external/usm/include/usm.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY external/spectra/include/Spectra
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY external/jk-tree/include/jk
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES data/camera_database.json
    DESTINATION ${CMAKE_INSTALL_DATADIR}/opencalibration)

install(EXPORT opencalibrationTargets
    NAMESPACE opencalibration::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/opencalibration)

configure_package_config_file(
    cmake/opencalibrationConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/opencalibrationConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/opencalibration)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/opencalibrationConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/opencalibrationConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/opencalibrationConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/opencalibration)

# ============= CPACK ================
set(CPACK_PACKAGE_NAME "opencalibration")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Open source camera calibration and orthophoto pipeline")
set(CPACK_PACKAGE_CONTACT "maintainer@opencalibration.org")

set(CPACK_DEBIAN_PACKAGE_DEPENDS
    "libeigen3-dev, libceres-dev, libopencv-dev, libgdal-dev, rapidjson-dev, libspdlog-dev, libtinyxml2-dev")
set(CPACK_DEBIAN_PACKAGE_SECTION "science")

set(CPACK_RPM_PACKAGE_REQUIRES
    "eigen3-devel, ceres-solver-devel, opencv-devel, gdal-devel, rapidjson-devel, spdlog-devel, tinyxml2-devel")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

include(CPack)
