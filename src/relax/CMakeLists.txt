add_library(oc_relax relax.cpp
                     relax_problem.cpp
                     relax_group.cpp
                     autodiff/decomposed_rotation.cpp
                     autodiff/pixel_error_orientation.cpp
                     autodiff/pixel_error_orientation_focal.cpp
                     autodiff/pixel_error_orientation_focal_radial.cpp
                     autodiff/pixel_error_orientation_focal_radial_tangential.cpp
                     autodiff/plane_intersection.cpp
                     autodiff/plane_intersection_focal_radial.cpp
                     autodiff/plane_intersection_nray.cpp
                     autodiff/plane_intersection_nray_focal_radial.cpp
                     autodiff/priors.cpp)
target_include_directories(oc_relax PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(oc_relax PUBLIC oc_geometry oc_distort oc_performance oc_surface ${CERES_LIBRARIES} spdlog::spdlog)
target_include_directories(oc_relax SYSTEM PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/jk-tree/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
