#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/surface_model.hpp>

namespace opencalibration::orthomosaic
{

struct OrthoMosaic
{
    cv::Mat pixelValues;
    cv::Mat overlap;
    cv::Mat cameraUUIDLsb;
};

OrthoMosaic generateOrthomosaic(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph);

} // namespace opencalibration::orthomosaic
