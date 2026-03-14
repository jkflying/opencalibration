add_library(oc_surface expand_mesh.cpp
                       refine_mesh.cpp
                       intersect.cpp
                       )
target_include_directories(oc_surface PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(oc_surface PUBLIC spdlog::spdlog
                                          oc_performance
                                          OpenMP::OpenMP_CXX)
target_include_directories(oc_surface SYSTEM PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/jk-tree/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
