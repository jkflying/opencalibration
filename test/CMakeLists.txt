find_package(GTest REQUIRED)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_data/LICENSE")
    message(ERROR "Test data not available, git submodule update --init --recursive to load")
endif()

add_executable(functional_tests gtest_main.cpp
                                test_camera_database.cpp
                                test_checkpoint.cpp
                                test_extract_features.cpp
                                test_extract_metadata.cpp
                                test_match.cpp
                                test_ortho_functional.cpp
                                test_pipeline.cpp
                                test_ransac_functional.cpp
                                test_refine_mesh.cpp
                                test_serialize_deserialize.cpp)

target_link_libraries(functional_tests PUBLIC gtest pthread oc_distort
                                                            oc_extract
                                                            oc_io
                                                            oc_match
                                                            oc_model_inliers
                                                            oc_ortho
                                                            oc_pipeline
                                                            oc_surface)

target_compile_definitions(functional_tests PUBLIC TEST_DATA_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/test_data/\"
                                                   TEST_DATA_OUTPUT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/test_output/\"
                                                   CAMERA_DATABASE_PATH=\"${CMAKE_BINARY_DIR}/camera_database.json\"
                                                   VISUAL_INSPECTION=1)

add_executable(unit_tests gtest_main.cpp
                          test_combinatorics.cpp
                          test_distort.cpp
                          test_geo_coord.cpp
                          test_geometry.cpp
                          test_graph.cpp
                          test_meshgraph.cpp
                          test_ortho.cpp
                          test_ransac_unit.cpp
                          test_relax.cpp
                          test_tree.cpp)

target_link_libraries(unit_tests PUBLIC gtest pthread oc_geo_coord
                                                      oc_geometry
                                                      oc_model_inliers
                                                      oc_ortho
                                                      oc_relax
                                                      oc_surface)

add_custom_target(test
                  USES_TERMINAL
                  COMMAND ${CMAKE_BINARY_DIR}/test/unit_tests
                  COMMAND ${CMAKE_BINARY_DIR}/test/functional_tests
                  COMMAND ${CMAKE_BINARY_DIR}/app/pipeline_runner -i "${CMAKE_CURRENT_SOURCE_DIR}/test_data/" -g test_data.geojson -d 4 -s test_data.json -p test_data.xyz
                  DEPENDS functional_tests unit_tests pipeline_runner)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Coverage")

    add_custom_target(clean_coverage
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${CMAKE_SOURCE_DIR}/external/venv/bin/fastcov -z
        COMMAND rm -r coverage_html || echo "no existing coverage report"
        COMMAND bash -c "echo `pwd`"
        DEPENDS functional_tests unit_tests pipeline_runner)

    add_dependencies(test clean_coverage)

    add_custom_target(test_coverage
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        USES_TERMINAL
        COMMAND ${CMAKE_SOURCE_DIR}/external/venv/bin/fastcov -i `realpath ${CMAKE_SOURCE_DIR}`  -e external/ -l -o clean_coverage.info
        COMMAND lcov --summary clean_coverage.info --ignore-errors mismatch
        DEPENDS test)

    add_custom_target(test_coverage_html
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        USES_TERMINAL
        COMMAND genhtml clean_coverage.info --output-directory coverage_html --branch-coverage --ignore-errors mismatch
        DEPENDS test_coverage)
endif()
