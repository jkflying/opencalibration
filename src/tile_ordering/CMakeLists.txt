add_library(oc_tile_ordering tile_ordering.cpp)
target_include_directories(oc_tile_ordering PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(oc_tile_ordering PUBLIC spdlog::spdlog)
