#include <opencalibration/relax/relax.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>

#include <ceres/ceres.h>

#include <unordered_set>

namespace opencalibration
{

void initializeOrientation(const std::vector<size_t> &node_ids, MeasurementGraph &graph)
{
    for (size_t node_id : node_ids)
    {
        MeasurementGraph::Node *node1 = graph.getNode(node_id);
        if (node1 == nullptr)
            return;

        // TODO: get connections, use relative orientations from RANSAC to initialize orientations
        node1->payload.orientation = Eigen::Quaterniond::Identity();
    }
}

} // namespace opencalibration
