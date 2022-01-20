#pragma once

#include <jk/KDTree.h>

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/relax_options.hpp>
#include <opencalibration/types/surface_model.hpp>

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
              const jk::tree::KDTree<size_t, 3> &imageGPSLocations, bool relax_all, bool disable_parallelism,
              const RelaxOptionSet &options);

    void trim_groups(size_t max_size);

    std::vector<std::function<void()>> get_runners(const MeasurementGraph &graph);

    std::vector<std::vector<size_t>> finalize(MeasurementGraph &graph);

    const std::vector<surface_model> &getSurfaceModels();

  private:
    std::vector<RelaxGroup> _groups;
    std::vector<surface_model> _surface_models, _previous_surface_models;

    std::unique_ptr<SpectralClustering<size_t, 3>> _k_groups;
};

} // namespace opencalibration
