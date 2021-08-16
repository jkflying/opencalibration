#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/relax_options.hpp>

#include <unordered_set>

namespace opencalibration
{
void relax(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
           std::unordered_map<size_t, CameraModel> &cam_models, const std::unordered_set<size_t> &edges_to_optimize,
           const RelaxOptionSet &options);

} // namespace opencalibration
