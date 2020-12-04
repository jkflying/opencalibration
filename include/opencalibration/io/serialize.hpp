#pragma once

#include <opencalibration/types/measurement_graph.hpp>

namespace opencalibration
{
std::string serialize(const MeasurementGraph &graph);

std::string toVisualizedGeoJson(const MeasurementGraph &graph,
                                std::function<Eigen::Vector3d(const Eigen::Vector3d &)> toGlobalCoordinates);
} // namespace opencalibration
