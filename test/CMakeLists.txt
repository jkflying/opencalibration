find_package(GTest REQUIRED)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_data/LICENSE")
    message(ERROR "Test data not available, git submodule update --init --recursive to load")
endif()

add_executable(functional_tests gtest_main.cpp
                                test_distort.cpp
                                test_extract_features.cpp
                                test_extract_metadata.cpp
                                test_match.cpp
                                test_pipeline.cpp
                                test_ransac_functional.cpp
                                test_serialize_deserialize.cpp)

target_link_libraries(functional_tests PUBLIC gtest pthread oc_distort
                                                            oc_extract
                                                            oc_io
                                                            oc_match
                                                            oc_model_inliers
                                                            oc_pipeline)

target_compile_definitions(functional_tests PUBLIC TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/test_data/\"
                                                   TEST_DATA_OUTPUT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/test_output/\"
                                                   VISUAL_INSPECTION=1)

add_executable(unit_tests gtest_main.cpp
                          test_geo_coord.cpp
                          test_graph.cpp
                          test_ransac_unit.cpp
                          test_relax.cpp
                          test_tree.cpp)

target_link_libraries(unit_tests PUBLIC gtest pthread oc_geo_coord
                                                      oc_model_inliers
                                                      oc_relax)

add_custom_target(test
                  COMMAND ${CMAKE_BINARY_DIR}/test/functional_tests
                  COMMAND ${CMAKE_BINARY_DIR}/test/unit_tests
                  COMMAND ${CMAKE_BINARY_DIR}/app/pipeline_runner "${CMAKE_CURRENT_SOURCE_DIR}/test_data/" 2> test_data.geojson
                  DEPENDS functional_tests unit_tests pipeline_runner)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Coverage")

    add_custom_target(clean_coverage
        COMMAND lcov --zerocounters --no-external --base-directory `realpath ${CMAKE_CURRENT_SOURCE_DIR}/..` --directory `realpath ${CMAKE_CURRENT_SOURCE_DIR}/..`
        COMMAND lcov --capture --initial --no-external --base-directory `realpath ${CMAKE_CURRENT_SOURCE_DIR}/..` --directory `realpath ${CMAKE_CURRENT_SOURCE_DIR}/..` --output-file base_coverage.info
        COMMAND rm -r coverage_html || echo "no existing coverage report"
        DEPENDS functional_tests unit_tests pipeline_runner)

    add_dependencies(test clean_coverage)

    add_custom_target(test_coverage
        COMMAND lcov --capture --base-directory `realpath ${CMAKE_BINARY_DIR}/..`  --directory `realpath ${CMAKE_BINARY_DIR}/..` --no-external --output-file test_coverage.info
        COMMAND lcov -a base_coverage.info -a test_coverage.info -o coverage.info
        COMMAND lcov -r coverage.info "*/TinyEXIF.*" -o clean_coverage.info
        COMMAND lcov --summary clean_coverage.info
        DEPENDS test)

    add_custom_target(test_coverage_html
        COMMAND genhtml clean_coverage.info --output-directory coverage_html --branch-coverage
        DEPENDS test_coverage)
endif()
