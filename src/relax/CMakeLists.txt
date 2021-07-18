add_library(oc_relax relax.cpp
                     relax_problem.cpp
                     relax_group.cpp
                     autodiff_cost_function.cpp
                     ceres_log_forwarding.cpp)
target_link_libraries(oc_relax PUBLIC oc_geometry oc_distort oc_performance ${CERES_LIBRARIES} spdlog::spdlog)
target_include_directories(oc_relax PUBLIC ${PROJECT_SOURCE_DIR}/external/jk-tree/include)
