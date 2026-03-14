add_library(oc_pipeline pipeline.cpp
                        load_stage.cpp
                        link_stage.cpp
                        relax_stage.cpp)
target_link_libraries(oc_pipeline PUBLIC oc_geo_coord
                                  PRIVATE oc_dense
                                          oc_extract
                                          oc_match
                                          oc_distort
                                          oc_model_inliers
                                          oc_relax
                                          oc_performance
                                          oc_ortho
                                          spdlog::spdlog)
target_include_directories(oc_pipeline PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/unordered_dense/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(oc_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}/external/jk-tree/include
    ${PROJECT_SOURCE_DIR}/external/usm/include)
