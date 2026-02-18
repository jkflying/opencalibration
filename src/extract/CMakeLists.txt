# TinyEXIF is third-party; compile it in its own target so clang-tidy is disabled for it.
add_library(tinyexif_obj OBJECT ${PROJECT_SOURCE_DIR}/external/TinyEXIF-master/TinyEXIF.cpp)
set_target_properties(tinyexif_obj PROPERTIES CXX_CLANG_TIDY "")
target_include_directories(tinyexif_obj SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/external/TinyEXIF-master)
target_link_libraries(tinyexif_obj PRIVATE tinyxml2)

add_library(oc_extract extract_features.cpp
                       extract_image.cpp
                       extract_metadata.cpp
                       camera_database.cpp
                       $<TARGET_OBJECTS:tinyexif_obj>)

target_include_directories(oc_extract SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/external/TinyEXIF-master
                                                     ${PROJECT_SOURCE_DIR}/external/jk-tree/include)

target_link_libraries(oc_extract PUBLIC opencv_core opencv_imgcodecs opencv_features2d spdlog::spdlog tinyxml2)

target_compile_definitions(oc_extract PRIVATE
    CAMERA_DATABASE_PATH="${CMAKE_BINARY_DIR}/camera_database.json")
