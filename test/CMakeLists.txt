find_package(GTest REQUIRED)

add_executable(functional_tests test_distort.cpp
                                test_extract.cpp
                                test_ransac_functional.cpp
                                test_match.cpp)

target_link_libraries(functional_tests PUBLIC gtest_main gtest pthread oc_distort
                                                                       oc_extract
                                                                       oc_match
                                                                       oc_model_inliers
                                                                       oc_pipeline)

target_compile_definitions(functional_tests PUBLIC TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/test_data/\"
                                                   TEST_DATA_OUTPUT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/test_output/\"
                                                   VISUAL_INSPECTION=1)

add_executable(unit_tests test_ransac_unit.cpp)

target_link_libraries(unit_tests PUBLIC gtest_main gtest pthread oc_model_inliers)

add_custom_target(test COMMAND ${CMAKE_BINARY_DIR}/test/functional_tests
                       COMMAND ${CMAKE_BINARY_DIR}/test/unit_tests
                  DEPENDS functional_tests unit_tests)
