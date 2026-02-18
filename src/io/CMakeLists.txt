# cv_raster_conversion.cpp uses cv::Mat::at<uint32_t> which clang rejects (no DataType<unsigned int>)
add_library(cv_raster_obj OBJECT cv_raster_conversion.cpp)
set_target_properties(cv_raster_obj PROPERTIES CXX_CLANG_TIDY "")
target_include_directories(cv_raster_obj SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(cv_raster_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_library(oc_io $<TARGET_OBJECTS:cv_raster_obj>
                  serialize_MeasurementGraph.cpp
                  serialize_MeshGraph.cpp
                  serialise_Raster.cpp
                  deserialize_MeasurementGraph.cpp
                  deserialize_MeshGraph.cpp
                  saveXYZ.cpp
                  checkpoint.cpp
                  base64.c)
target_link_libraries(oc_io PUBLIC spdlog::spdlog)
