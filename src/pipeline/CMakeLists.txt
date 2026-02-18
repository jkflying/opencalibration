add_library(oc_pipeline pipeline.cpp
                        load_stage.cpp
                        link_stage.cpp
                        relax_stage.cpp)
target_link_libraries(oc_pipeline PUBLIC oc_extract
                                          oc_match
                                          oc_distort
                                          oc_model_inliers
                                          oc_geo_coord
                                          oc_relax
                                          oc_performance
                                          oc_ortho
                                          spdlog::spdlog)
target_include_directories(oc_pipeline SYSTEM PUBLIC ${PROJECT_SOURCE_DIR}/external/jk-tree/include
                                                    ${PROJECT_SOURCE_DIR}/external/usm/include)
