add_library(oc_surface expand_mesh.cpp
                       refine_mesh.cpp
                       intersect.cpp
                       )
target_link_libraries(oc_surface PUBLIC spdlog::spdlog
                                          oc_performance
                                          OpenMP::OpenMP_CXX)
target_include_directories(oc_surface SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/external/jk-tree/include)
