add_library(oc_ortho ortho.cpp image_cache.cpp color_balance.cpp blending.cpp)
target_include_directories(oc_ortho PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(oc_ortho PUBLIC spdlog::spdlog
                                      oc_performance
                                      OpenMP::OpenMP_CXX
                                      ${GDAL_LIBRARIES}
                                      ${OpenCV_LIBS}
                                      ${CERES_LIBRARIES}
)
target_include_directories(oc_ortho SYSTEM PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/jk-tree/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(oc_ortho SYSTEM PRIVATE ${GDAL_INCLUDE_DIRS})
