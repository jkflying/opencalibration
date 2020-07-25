add_library(oc_relax relax.cpp)
target_link_libraries(oc_pipeline PRIVATE ${CERES_LIBRARIES})
