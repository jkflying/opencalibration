add_library(oc_geometry intersection.cpp)
target_link_libraries(oc_geometry PUBLIC spdlog::spdlog)
target_include_directories(oc_geometry PUBLIC ${PROJECT_SOURCE_DIR}/external/spectra-0.9.0/include)
