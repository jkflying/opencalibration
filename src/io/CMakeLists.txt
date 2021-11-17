add_library(oc_io serialize_MeasurementGraph.cpp
                  serialize_MeshGraph.cpp
                  deserialize_MeasurementGraph.cpp
                  deserialize_MeshGraph.cpp
                  saveXYZ.cpp
                  base64.c)
target_link_libraries(oc_io PUBLIC spdlog::spdlog)
