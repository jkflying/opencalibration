add_library(oc_ortho ortho.cpp image_cache.cpp color_balance.cpp blending.cpp)
target_link_libraries(oc_ortho PUBLIC spdlog::spdlog
                                      oc_performance
                                      OpenMP::OpenMP_CXX
                                      ${GDAL_LIBRARIES}
                                      ${OpenCV_LIBS}
                                      ${CERES_LIBRARIES}
)
target_include_directories(oc_ortho PRIVATE ${PROJECT_SOURCE_DIR}/external/jk-tree/include
                                            ${GDAL_INCLUDE_DIRS})
