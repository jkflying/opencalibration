#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/mesh_graph.hpp>

#include <iosfwd>

namespace opencalibration
{
bool serialize(const MeasurementGraph &graph, std::ostream &out);

bool serialize(const MeshGraph &graph, std::ostream &out);

std::string serializeNode(const MeasurementGraph &graph, size_t node_id);
std::string serializeEdge(const MeasurementGraph &graph, size_t edge_id);

// Each triangle once, ascending node ids, orientation normalised, sorted
std::vector<std::array<size_t, 3>> meshFaces(const MeshGraph &graph);

bool toVisualizedGeoJson(const MeasurementGraph &graph,
                         std::function<Eigen::Vector3d(const Eigen::Vector3d &)> toGlobalCoordinates,
                         std::ostream &out);

bool toVisualizedGeoJson(const MeasurementGraph &graph, const std::vector<size_t> &node_ids,
                         const std::vector<size_t> &edge_ids,
                         std::function<Eigen::Vector3d(const Eigen::Vector3d &)> toGlobalCoordinates,
                         std::ostream &out);

} // namespace opencalibration
