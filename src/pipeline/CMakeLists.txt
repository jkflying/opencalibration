add_library(oc_pipeline pipeline.cpp)
target_link_libraries(oc_pipeline PRIVATE oc_extract
                                          oc_match
                                          oc_distort
                                          oc_model_inliers
                                          oc_geo_coord
                                          oc_relax
                                          OpenMP::OpenMP_CXX)
