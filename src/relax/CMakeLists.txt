add_library(oc_relax relax.cpp)
target_link_libraries(oc_relax PUBLIC ${CERES_LIBRARIES} spdlog::spdlog)
