add_library(oc_dense dense_stereo.cpp)
target_link_libraries(oc_dense PUBLIC spdlog::spdlog
                                       oc_surface
                                       oc_distort
                                       OpenMP::OpenMP_CXX)
target_include_directories(oc_dense SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/external/jk-tree/include)
