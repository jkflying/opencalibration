#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>

#include <unordered_set>

namespace opencalibration
{

void initializeOrientation(const MeasurementGraph &graph, std::vector<NodePose> &nodes);
void relaxDecompositions(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                         const std::unordered_set<size_t> &edges);

// TODO: keep the ray intersection points?
// void relaxMeasurements(const MeasurementGraph &graph, std::vector<NodePose> &nodes);

} // namespace opencalibration
