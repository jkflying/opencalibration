#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>

#include <unordered_set>

namespace opencalibration
{

void initializeOrientation(const MeasurementGraph &graph, std::vector<NodePose> &nodes);
void relaxDecompositions(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                         const std::unordered_set<size_t> &edges_to_optimize);

// TODO: keep the ray intersection points somewhere?
void relaxMeasurements(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                       const std::unordered_set<size_t> &edges_to_optimize);

} // namespace opencalibration
