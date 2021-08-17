#pragma once

#include <jk/KDTree.h>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_links.hpp>

#include <mutex>
#include <vector>

namespace opencalibration
{

class LinkStage
{
    struct edge_payload
    {
        size_t loop_index; // used for re-ordering multithreading component

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
    std::vector<edge_payload> _all_inlier_measurements;
    std::mutex _measurement_mutex;

    std::vector<NodeLinks> _links;
};

} // namespace opencalibration
