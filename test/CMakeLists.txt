find_package(GTest REQUIRED)

add_executable(functional_tests test_extract.cpp)
target_link_libraries(functional_tests PUBLIC gtest_main gtest pthread oc_extract)

target_compile_definitions(functional_tests PUBLIC TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/test_data/\"
                                                   TEST_DATA_OUTPUT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/test_output/\")
