add_executable(pipeline_runner pipeline_runner.cpp)

target_link_libraries(pipeline_runner PRIVATE oc_pipeline oc_io oc_extract stdc++fs)
target_include_directories(pipeline_runner PRIVATE ${CMAKE_SOURCE_DIR}/external/commandline)

add_executable(extract_camera_db extract_camera_db.cpp)
target_link_libraries(extract_camera_db PRIVATE oc_io oc_extract stdc++fs)
target_include_directories(extract_camera_db PRIVATE ${CMAKE_SOURCE_DIR}/external/commandline)
