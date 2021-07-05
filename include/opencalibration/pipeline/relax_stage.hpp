#pragma once

#include <jk/KDTree.h>

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>

#include <functional>
#include <vector>

namespace opencalibration
{
class RelaxGroup;

template <typename, size_t> class SpectralClustering;

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
    std::unique_ptr<SpectralClustering<size_t, 3>> _k_groups;
};

} // namespace opencalibration
