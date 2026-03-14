add_library(oc_model_inliers ransac.cpp
                             homography_model.cpp
                             fundamental_matrix_model.cpp
                             essential_matrix_model.cpp)
target_include_directories(oc_model_inliers PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(oc_model_inliers PUBLIC opencv_calib3d spdlog::spdlog)
