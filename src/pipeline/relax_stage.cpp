#include <opencalibration/pipeline/relax_stage.hpp>

#include <spdlog/spdlog.h>

namespace opencalibration
{

void RelaxStage::init(const MeasurementGraph &graph, const std::vector<size_t> &node_ids)
{

    spdlog::info("Queueing {} image nodes for graph relaxation", node_ids.size());
    _local_poses.clear();
    _local_poses.reserve(node_ids.size());
    for (size_t node_id : node_ids)
    {
        const auto *node = graph.getNode(node_id);
        CameraPose pose;
        pose.node_id = node_id;
        pose.orientation = node->payload.orientation;
        pose.position = node->payload.position;
        _local_poses.push_back(pose);
    }
}

std::vector<std::function<void()>> RelaxStage::get_runners(const MeasurementGraph &graph)
{
    struct measurement
    {
        CameraPose *pose;
    };

    auto runner = [&]() {
        std::vector<measurement> measurements;
        measurements.reserve(_local_poses.size() * 10);
        for (auto &pose : _local_poses)
        {
            auto *node = graph.getNode(pose.node_id);
            const auto &edges = node->getEdges();
            for (size_t edge_id : edges)
            {
                auto *edge = graph.getEdge(edge_id);

                measurement m;
                m.pose = &pose;
                if (edge->getDest() == pose.node_id)
                {
                }
                else if (edge->getSource() == pose.node_id)
                {
                }
                measurements.push_back(m);
            }
        }
    };

    return {runner};
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
