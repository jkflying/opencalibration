#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/relax_options.hpp>

#include <unordered_set>

namespace opencalibration
{

void initializeOrientation(const MeasurementGraph &graph, std::vector<NodePose> &nodes);

void relax(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
           const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options);

[[deprecated]] void relaxDecompositions(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                        const std::unordered_set<size_t> &edges_to_optimize);

[[deprecated]] void relax3dPointMeasurements(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                             const std::unordered_set<size_t> &edges_to_optimize);

[[deprecated]] void relaxGroundPlaneMeasurements(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                                 const std::unordered_set<size_t> &edges_to_optimize);

} // namespace opencalibration
