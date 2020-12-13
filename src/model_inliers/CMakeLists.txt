add_library(oc_model_inliers ransac.cpp)
target_link_libraries(oc_model_inliers PUBLIC opencv_calib3d spdlog::spdlog)
