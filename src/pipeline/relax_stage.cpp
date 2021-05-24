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
                      const jk::tree::KDTree<size_t, 3> &imageGPSLocations, bool force_optimize_all)
{
    _directly_connected.clear();
    _edges_to_optimize.clear();
    _ids_added.clear();
    _local_poses.clear();

    int connection_order_to_include = 1;

    _local_poses.reserve(node_ids.size());

    _optimize_all = force_optimize_all || graph.size_nodes() > 1.5 * _last_graph_size_full_relax;
    if (_optimize_all)
    {
        _last_graph_size_full_relax = graph.size_nodes();
    }

    for (size_t node_id : node_ids)
    {
        const auto *node = graph.getNode(node_id);
        NodePose pose;
        pose.node_id = node_id;
        pose.orientation = node->payload.orientation;
        pose.position = node->payload.position;
        _local_poses.push_back(pose);
        build_optimization_edges(graph, imageGPSLocations, pose.node_id);
    }

    for (int i = 0; i < connection_order_to_include; i++)
    {
        // remove already added node ids from nodes to still connect
        auto newly_connected = _directly_connected;
        for (size_t id : _ids_added)
        {
            newly_connected.erase(id);
        }

        _local_poses.reserve(_local_poses.size() + newly_connected.size());

        // add nodes that are directly connected
        for (size_t node_id : newly_connected)
        {
            const auto *node = graph.getNode(node_id);
            NodePose pose;
            pose.node_id = node_id;
            pose.orientation = node->payload.orientation;
            pose.position = node->payload.position;
            _local_poses.push_back(pose);
            build_optimization_edges(graph, imageGPSLocations, pose.node_id);
        }
    }

    if (_optimize_all)
    {
        for (auto iter = graph.nodebegin(); iter != graph.nodeend(); ++iter)
        {
            if (_ids_added.find(iter->first) != _ids_added.end())
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
            build_optimization_edges(graph, imageGPSLocations, pose.node_id);
        }
    }

    initializeOrientation(graph, _local_poses);

    spdlog::info("Queueing {} image nodes, {} edges for graph relaxation", _local_poses.size(),
                 _edges_to_optimize.size());
}

void RelaxStage::build_optimization_edges(const MeasurementGraph &graph,
                                          const jk::tree::KDTree<size_t, 3> &imageGPSLocations, size_t node_id)
{

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
        if (edge->getSource() == node_id &&
            ideally_connected_nodes.find(edge->getDest()) != ideally_connected_nodes.end())
        {
            _edges_to_optimize.insert(edge_id);
            _directly_connected.insert(edge->getDest());
        }
        else if (edge->getDest() == node_id &&
                 ideally_connected_nodes.find(edge->getSource()) != ideally_connected_nodes.end())
        {
            _edges_to_optimize.insert(edge_id);
            _directly_connected.insert(edge->getSource());
        }
    }
    _ids_added.insert(node_id);
}

std::vector<std::function<void()>> RelaxStage::get_runners(const MeasurementGraph &graph)
{
    if (_local_poses.size() > 0)
    {
        return {[&]() {
            //             if (_optimize_all)
            //             {
            //                 relaxGroundPlaneMeasurements(graph, _local_poses, _edges_to_optimize);
            //             }
            //             else
            //             {
            relaxDecompositions(graph, _local_poses, _edges_to_optimize);
            //             }
        }};
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
