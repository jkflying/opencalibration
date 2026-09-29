add_library(oc_io cv_raster_conversion.cpp
                  serialize_MeasurementGraph.cpp
                  serialize_MeshGraph.cpp
                  deserialize_MeasurementGraph.cpp
                  deserialize_MeshGraph.cpp
                  saveXYZ.cpp
                  checkpoint.cpp
                  base64.c)
target_include_directories(oc_io PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(oc_io SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(oc_io PUBLIC spdlog::spdlog PRIVATE zstd)
