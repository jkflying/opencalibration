#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/relax_options.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <ankerl/unordered_dense.h>

namespace opencalibration
{
surface_model relax(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                    ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                    const ankerl::unordered_dense::set<size_t> &edges_to_optimize, const RelaxOptionSet &options,
                    const std::vector<surface_model> &previousSurfaces);

} // namespace opencalibration
