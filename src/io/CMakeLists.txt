add_library(oc_io serialize.cpp
                  deserialize.cpp
                  base64.c)
target_link_libraries(oc_io PUBLIC spdlog::spdlog)
