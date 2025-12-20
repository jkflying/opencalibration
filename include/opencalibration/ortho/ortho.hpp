#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/raster.hpp>
#include <opencalibration/types/surface_model.hpp>

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
double calculateGSD(const MeasurementGraph &graph, double mean_surface_z);

OrthoMosaic generateOrthomosaic(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph);

} // namespace opencalibration::orthomosaic
