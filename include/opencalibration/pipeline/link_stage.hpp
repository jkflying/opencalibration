#pragma once

#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/types/node_links.hpp>

namespace opencalibration
{

class BuildLinksStage
{
    struct inlier_measurement
    {
        size_t node_id;
        size_t match_node_id;
        camera_relations relations;
    };

  public:
    void init(const MeasurementGraph &graph, const jk::tree::KDTree<size_t, 3> &imageGPSLocations,
              const std::vector<size_t> &node_ids);

    std::vector<std::function<void()>> get_runners(const MeasurementGraph &graph);

    std::vector<size_t> finalize(MeasurementGraph &graph);

  private:
    std::vector<inlier_measurement> _all_inlier_measurements;
    std::mutex _measurement_mutex;

    std::vector<NodeLinks> _links;
};

} // namespace opencalibration
