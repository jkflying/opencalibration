add_executable(pipeline_runner pipeline_runner.cpp)

target_link_libraries(pipeline_runner PRIVATE oc_pipeline oc_io stdc++fs)
target_include_directories(pipeline_runner PRIVATE ${CMAKE_SOURCE_DIR}/external/commandline)
