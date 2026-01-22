add_library(oc_io cv_raster_conversion.cpp
                  serialize_MeasurementGraph.cpp
                  serialize_MeshGraph.cpp
                  serialise_Raster.cpp
                  deserialize_MeasurementGraph.cpp
                  deserialize_MeshGraph.cpp
                  saveXYZ.cpp
                  checkpoint.cpp
                  base64.c)
target_link_libraries(oc_io PUBLIC spdlog::spdlog)
