#pragma once

#include <opencalibration/types/measurement_graph.hpp>

#include <iosfwd>

namespace opencalibration
{
bool serialize(const MeasurementGraph &graph, std::ostream &out);

bool toVisualizedGeoJson(const MeasurementGraph &graph,
                         std::function<Eigen::Vector3d(const Eigen::Vector3d &)> toGlobalCoordinates,
                         std::ostream &out);

bool toVisualizedGeoJson(const MeasurementGraph &graph, const std::vector<size_t> &node_ids,
                         const std::vector<size_t> &edge_ids,
                         std::function<Eigen::Vector3d(const Eigen::Vector3d &)> toGlobalCoordinates,
                         std::ostream &out);
} // namespace opencalibration
