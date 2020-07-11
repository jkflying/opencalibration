
find_package(GTest REQUIRED)

add_executable(unittests test_extract.cpp)
target_link_libraries(unittests PUBLIC gtest_main gtest pthread oc_extract)

