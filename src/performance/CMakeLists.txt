add_library(oc_performance performance.cpp)
target_include_directories(oc_performance PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(oc_performance PUBLIC spdlog::spdlog)

