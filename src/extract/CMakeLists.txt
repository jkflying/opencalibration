add_library(oc_extract extract_features.cpp
                       extract_image.cpp
                       extract_metadata.cpp
                       camera_database.cpp
                       ${PROJECT_SOURCE_DIR}/external/TinyEXIF-master/TinyEXIF.cpp)

target_include_directories(oc_extract PRIVATE ${PROJECT_SOURCE_DIR}/external/TinyEXIF-master
                                              ${PROJECT_SOURCE_DIR}/external/jk-tree/include)

target_link_libraries(oc_extract PUBLIC opencv_core opencv_imgcodecs opencv_features2d spdlog::spdlog tinyxml2)

target_compile_definitions(oc_extract PRIVATE
    CAMERA_DATABASE_PATH="${CMAKE_BINARY_DIR}/camera_database.json")
