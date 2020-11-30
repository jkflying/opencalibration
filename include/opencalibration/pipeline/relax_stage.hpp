#pragma once

#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/types/node_pose.hpp>

namespace opencalibration
{

class RelaxStage
{
  public:
    void init(const MeasurementGraph &graph, const std::vector<size_t> &node_ids);

    std::vector<std::function<void()>> get_runners(const MeasurementGraph &graph);

    std::vector<size_t> finalize(MeasurementGraph &graph);

  private:

    std::vector<NodePose> _local_poses;
};

} // namespace opencalibration
