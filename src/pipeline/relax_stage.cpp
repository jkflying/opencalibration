#include <opencalibration/pipeline/relax_stage.hpp>

#include <opencalibration/relax/relax.hpp>

#include <spdlog/spdlog.h>

#include <unordered_set>

namespace opencalibration
{

void RelaxStage::init(const MeasurementGraph &graph, const std::vector<size_t> &node_ids, bool optimize_all)
{

    spdlog::info("Queueing {} image nodes for graph relaxation", node_ids.size());
    _local_poses.clear();

    _local_poses.reserve(node_ids.size());
    for (size_t node_id : node_ids)
    {
        const auto *node = graph.getNode(node_id);
        NodePose pose;
        pose.node_id = node_id;
        pose.orientation = node->payload.orientation;
        pose.position = node->payload.position;
        _local_poses.push_back(pose);
    }

    initializeOrientation(graph, _local_poses);

    if (optimize_all)
    {
        std::unordered_set<size_t> ids_added;
        for (size_t node_id : node_ids)
        {
            ids_added.insert(node_id);
        }
        for (auto iter = graph.nodebegin(); iter != graph.nodeend(); ++iter)
        {
            if (ids_added.find(iter->first) != ids_added.end())
            {
                continue;
            }
            if (iter->second.payload.orientation.coeffs().hasNaN() || iter->second.payload.position.hasNaN())
            {
                continue;
            }
            NodePose pose;
            pose.node_id = iter->first;
            pose.orientation = iter->second.payload.orientation;
            pose.position = iter->second.payload.position;
            _local_poses.push_back(pose);
        }
    }
}

std::vector<std::function<void()>> RelaxStage::get_runners(const MeasurementGraph &graph)
{
    if (_local_poses.size() > 0)
    {
        return {[&]() { relaxDecompositions(graph, _local_poses); }};
    }
    return {};
}

std::vector<size_t> RelaxStage::finalize(MeasurementGraph &graph)
{
    for (const auto &pose : _local_poses)
    {
        auto *node = graph.getNode(pose.node_id);
        node->payload.orientation = pose.orientation;
        node->payload.position = pose.position;
    }
    _local_poses.clear();
    return {};
}

} // namespace opencalibration
