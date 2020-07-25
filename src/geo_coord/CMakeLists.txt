add_library(oc_geo_coord geo_coord.cpp)
target_link_libraries(oc_geo_coord PRIVATE ${GDAL_LIBRARIES})
