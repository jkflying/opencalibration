add_library(oc_relax relax.cpp)
target_link_libraries(oc_pipeline PUBLIC ${CERES_LIBRARIES})
