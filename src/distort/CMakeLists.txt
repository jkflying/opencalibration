add_library(oc_distort distort_keypoints.cpp invert_distortion.cpp)
target_include_directories(oc_distort PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
