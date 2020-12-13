add_library(oc_extract extract_features.cpp
                       extract_metadata.cpp
                       TinyEXIF.cpp
                       tinyxml2.cpp)

target_link_libraries(oc_extract PUBLIC opencv_core opencv_imgcodecs opencv_features2d spdlog::spdlog)
