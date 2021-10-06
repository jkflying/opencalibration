add_library(oc_model_inliers ransac.cpp
                             homography_model.cpp
                             fundamental_matrix_model.cpp)
target_link_libraries(oc_model_inliers PUBLIC opencv_calib3d spdlog::spdlog)
