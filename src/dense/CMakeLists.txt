add_library(oc_dense dense_stereo.cpp)
target_link_libraries(oc_dense PUBLIC spdlog::spdlog
                                       oc_surface)
