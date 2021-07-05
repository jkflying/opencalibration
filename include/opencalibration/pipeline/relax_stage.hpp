#pragma once

#include <jk/KDTree.h>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>

#include <functional>
#include <vector>

namespace opencalibration
{
class RelaxGroup;

class RelaxStage
{
  public:
    RelaxStage();
    ~RelaxStage();
    void init(const MeasurementGraph &graph, const std::vector<size_t> &node_ids,
              const jk::tree::KDTree<size_t, 3> &imageGPSLocations, bool final_global_relax);

    std::vector<std::function<void()>> get_runners(const MeasurementGraph &graph);

    std::vector<size_t> finalize(MeasurementGraph &graph);

  private:
    std::vector<RelaxGroup> _groups;
};

} // namespace opencalibration
