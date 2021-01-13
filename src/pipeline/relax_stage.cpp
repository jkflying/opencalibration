#include <opencalibration/pipeline/relax_stage.hpp>

#include <opencalibration/relax/relax.hpp>

#include <spdlog/spdlog.h>

#include <unordered_set>

namespace
{

std::array<double, 3> to_array(const Eigen::Vector3d &v)
{
    return {v.x(), v.y(), v.z()};
}
} // namespace

namespace opencalibration
{

void RelaxStage::init(const MeasurementGraph &graph, const std::vector<size_t> &node_ids,
                      const jk::tree::KDTree<size_t, 3> &imageGPSLocations, bool optimize_all)
{
    _local_poses.clear();
    _edges_to_optimize.clear();

    _local_poses.reserve(node_ids.size());

    // TODO: prune extra graph edges that don't make sense anymore due to higher risk of false matches
    auto build_optimization_edges = [&](size_t node_id) {
        const auto *node = graph.getNode(node_id);
        auto knn = imageGPSLocations.searchKnn(to_array(node->payload.position), 10);
        std::unordered_set<size_t> ideally_connected_nodes(knn.size());
        for (const auto &edge : knn)
        {
            ideally_connected_nodes.insert(edge.payload);
        }
        ideally_connected_nodes.erase(node_id);
        for (size_t edge_id : node->getEdges())
        {
            const auto *edge = graph.getEdge(edge_id);
            if ((edge->getSource() == node_id &&
                 ideally_connected_nodes.find(edge->getDest()) != ideally_connected_nodes.end()) ||
                (edge->getDest() == node_id &&
                 ideally_connected_nodes.find(edge->getSource()) != ideally_connected_nodes.end()))
            {
                _edges_to_optimize.insert(edge_id);
            }
        }
    };

    for (size_t node_id : node_ids)
    {
        const auto *node = graph.getNode(node_id);
        NodePose pose;
        pose.node_id = node_id;
        pose.orientation = node->payload.orientation;
        pose.position = node->payload.position;
        _local_poses.push_back(pose);
        build_optimization_edges(pose.node_id);
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
            build_optimization_edges(pose.node_id);
        }
    }
    spdlog::info("Queueing {} image nodes, {} edges for graph relaxation", _local_poses.size(),
                 _edges_to_optimize.size());
}

std::vector<std::function<void()>> RelaxStage::get_runners(const MeasurementGraph &graph)
{
    if (_local_poses.size() > 0)
    {
        return {[&]() { relaxDecompositions(graph, _local_poses, _edges_to_optimize); }};
    }
    return {};
}

std::vector<size_t> RelaxStage::finalize(MeasurementGraph &graph)
{
    std::vector<size_t> optimized_ids;
    optimized_ids.reserve(_local_poses.size());
    for (const auto &pose : _local_poses)
    {
        auto *node = graph.getNode(pose.node_id);
        node->payload.orientation = pose.orientation;
        node->payload.position = pose.position;
        optimized_ids.push_back(pose.node_id);
    }
    _local_poses.clear();
    return optimized_ids;
}

} // namespace opencalibration
