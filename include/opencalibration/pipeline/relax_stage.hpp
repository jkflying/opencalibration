#pragma once

#include <opencalibration/pipeline/pipeline.hpp>
#include <opencalibration/types/node_pose.hpp>

#include <unordered_set>

namespace opencalibration
{

class RelaxStage
{
  public:
    void init(const MeasurementGraph &graph, const std::vector<size_t> &node_ids,
              const jk::tree::KDTree<size_t, 3> &imageGPSLocations, bool force_optimize_all);

    std::vector<std::function<void()>> get_runners(const MeasurementGraph &graph);

    std::vector<size_t> finalize(MeasurementGraph &graph);

  private:
    std::vector<NodePose> _local_poses;
    std::unordered_set<size_t> _edges_to_optimize;

    size_t _last_graph_size_full_relax = 0;
};

} // namespace opencalibration
