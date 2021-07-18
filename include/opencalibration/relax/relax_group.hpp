#pragma once

#include <jk/KDTree.h>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/relax_options.hpp>

#include <unordered_set>

namespace opencalibration
{

class RelaxGroup
{
  public:
    void init(const MeasurementGraph &graph, const std::vector<size_t> &node_ids,
              const jk::tree::KDTree<size_t, 3> &imageGPSLocations, size_t graph_connection_depth,
              const RelaxOptionSet &options);

    void run(const MeasurementGraph &graph);

    std::vector<size_t> finalize(MeasurementGraph &graph);

  private:
    std::vector<NodePose> _local_poses;
    std::unordered_set<size_t> _edges_to_optimize;
    std::unordered_set<size_t> _nodes_to_optimize;
    std::unordered_set<size_t> _directly_connected;

    void build_optimization_edges(const MeasurementGraph &graph, const jk::tree::KDTree<size_t, 3> &imageGPSLocations,
                                  size_t node_id);

    RelaxOptionSet _relax_options;
};

} // namespace opencalibration
