set(EXPECTED_FILES
    test_data.geojson
    test_data.json
    test_data.xyz
    test_data.ply
    test_data_ortho.tif
    test_data_dsm.tif
    test_data_textured.obj
    test_data_textured.mtl
    test_data_textured.jpg
    test_data_thumb.png
    test_data_source.png
    test_data_overlap.png
    test_checkpoint/graph.json
    test_checkpoint/metadata.json
)

set(ALL_PASSED TRUE)
foreach(F ${EXPECTED_FILES})
    set(FULL_PATH "${TEST_DIR}/${F}")
    if(NOT EXISTS "${FULL_PATH}")
        message(WARNING "Missing expected output: ${F}")
        set(ALL_PASSED FALSE)
    else()
        file(SIZE "${FULL_PATH}" FSIZE)
        if(FSIZE EQUAL 0)
            message(WARNING "Empty output file: ${F}")
            set(ALL_PASSED FALSE)
        endif()
    endif()
endforeach()

if(NOT ALL_PASSED)
    message(FATAL_ERROR "Pipeline output verification failed")
else()
    message(STATUS "All pipeline outputs verified")
endif()
