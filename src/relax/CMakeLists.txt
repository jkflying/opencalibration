add_library(oc_relax relax.cpp)
target_link_libraries(oc_relax PUBLIC oc_geometry oc_distort ${CERES_LIBRARIES} spdlog::spdlog)
