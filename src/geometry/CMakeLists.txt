add_library(oc_geometry intersection.cpp)
target_link_libraries(oc_geometry PUBLIC spdlog::spdlog oc_distort)
target_include_directories(oc_geometry PUBLIC ${PROJECT_SOURCE_DIR}/external/spectra/include)
