#pragma once

#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/types/node_links.hpp>

namespace opencalibration
{

class RelaxStage
{
  public:
    void init(const MeasurementGraph &graph, const std::vector<size_t> &node_ids);

    std::vector<std::function<void()>> get_runners(const MeasurementGraph &graph);

    std::vector<size_t> finalize(MeasurementGraph &graph);

  private:
    struct CameraPose
    {
        size_t node_id;
        Eigen::Quaterniond orientation;
        Eigen::Vector3d position;
    };

    std::vector<CameraPose> _local_poses;
};

} // namespace opencalibration
