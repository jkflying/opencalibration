find_package(GTest REQUIRED)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_data/LICENSE")
    message(ERROR "Test data not available, git submodule update --init --recursive to load")
endif()

add_executable(functional_tests gtest_main.cpp
                                test_distort.cpp
                                test_extract_features.cpp
                                test_extract_metadata.cpp
                                test_pipeline.cpp
                                test_ransac_functional.cpp
                                test_match.cpp
                                test_serialize_deserialize.cpp)

target_link_libraries(functional_tests PUBLIC gtest pthread oc_distort
                                                                       oc_extract
                                                                       oc_match
                                                                       oc_model_inliers
                                                                       oc_pipeline
                                                                       oc_io)

target_compile_definitions(functional_tests PUBLIC TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/test_data/\"
                                                   TEST_DATA_OUTPUT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/test_output/\"
                                                   VISUAL_INSPECTION=1)

add_executable(unit_tests gtest_main.cpp
                          test_ransac_unit.cpp
                          test_geo_coord.cpp
                          test_graph.cpp
                          test_tree.cpp)

target_link_libraries(unit_tests PUBLIC gtest pthread oc_model_inliers oc_geo_coord)

add_custom_target(test COMMAND ${CMAKE_BINARY_DIR}/test/functional_tests
                       COMMAND ${CMAKE_BINARY_DIR}/test/unit_tests
                  DEPENDS functional_tests unit_tests)
