add_library(oc_geo_coord geo_coord.cpp)
target_link_libraries(oc_geo_coord PUBLIC ${GDAL_LIBRARIES})
