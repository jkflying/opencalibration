add_library(oc_extract extract_features.cpp
                       extract_metadata.cpp
                       ${PROJECT_SOURCE_DIR}/external/TinyEXIF-master/TinyEXIF.cpp)

target_include_directories(oc_extract PRIVATE ${PROJECT_SOURCE_DIR}/external/TinyEXIF-master
                                              ${PROJECT_SOURCE_DIR}/external/jk-tree/include)

target_link_libraries(oc_extract PUBLIC opencv_core opencv_imgcodecs opencv_features2d spdlog::spdlog tinyxml2)
