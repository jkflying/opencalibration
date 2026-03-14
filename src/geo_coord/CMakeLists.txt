add_library(oc_geo_coord geo_coord.cpp)
target_include_directories(oc_geo_coord PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(oc_geo_coord PUBLIC ${GDAL_LIBRARIES}
                                          spdlog::spdlog)
