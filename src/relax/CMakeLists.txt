add_library(oc_relax relax.cpp
                     relax_problem.cpp
                     relax_group.cpp
                     autodiff_cost_function.cpp)
target_include_directories(oc_relax PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(oc_relax PUBLIC oc_geometry oc_distort oc_performance oc_surface ${CERES_LIBRARIES} spdlog::spdlog)
target_include_directories(oc_relax SYSTEM PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/jk-tree/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
