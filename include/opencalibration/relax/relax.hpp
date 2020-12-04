#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>

namespace opencalibration
{

void initializeOrientation(const std::vector<size_t> &node_ids, MeasurementGraph &graph);

void relaxDecompositions(const MeasurementGraph &graph, std::vector<NodePose> &nodes);

// TODO: keep the ray intersection points?
void relaxMeasurements(const MeasurementGraph &graph, std::vector<NodePose> &nodes);

} // namespace opencalibration
