#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/raster.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <string>

namespace opencalibration
{
class GeoCoord;
}

namespace opencalibration::orthomosaic
{

struct OrthoMosaic
{
    GenericRaster pixelValues;
    RasterLayer<float> dsm;
    RasterLayer<uint8_t> overlap;
    RasterLayer<uint32_t> cameraUUID;
    double gsd;
};

struct OrthoMosaicBounds
{
    double min_x, max_x, min_y, max_y;
    double mean_surface_z;
};

OrthoMosaicBounds calculateBoundsAndMeanZ(const std::vector<surface_model> &surfaces);
double calculateGSD(const MeasurementGraph &graph, const std::unordered_set<size_t> &involved_nodes,
                    double mean_surface_z);

OrthoMosaic generateOrthomosaic(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph);

void generateGeoTIFFOrthomosaic(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                                const opencalibration::GeoCoord &coord_system, const std::string &output_path,
                                int tile_size = 1024);

} // namespace opencalibration::orthomosaic
