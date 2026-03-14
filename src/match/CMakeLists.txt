add_library(oc_match match_features.cpp)
target_include_directories(oc_match PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(oc_match PUBLIC spdlog::spdlog)
target_include_directories(oc_match PRIVATE ${PROJECT_SOURCE_DIR}/external/jk-tree/include)
