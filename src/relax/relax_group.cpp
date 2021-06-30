#include <opencalibration/relax/relax_group.hpp>

#include <opencalibration/performance/performance.hpp>
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

void RelaxGroup::init(const MeasurementGraph &graph, const std::vector<size_t> &node_ids,
                      const jk::tree::KDTree<size_t, 3> &imageGPSLocations, size_t graph_connection_depth,
                      RelaxType relax_type)
{
    _directly_connected.clear();
    _edges_to_optimize.clear();
    _nodes_to_optimize.clear();
    _local_poses.clear();

    _relax_type = relax_type;
    _local_poses.reserve(node_ids.size());

    _nodes_to_optimize.insert(node_ids.begin(), node_ids.end());

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

    for (size_t i = 0; i < graph_connection_depth; i++)
    {
        // remove already added node ids from nodes to still connect
        std::unordered_set<size_t> newly_connected;
        for (size_t id : _directly_connected)
        {
            if (_nodes_to_optimize.find(id) == _nodes_to_optimize.end())
            {
                newly_connected.insert(id);
            }
        }

        _local_poses.reserve(_local_poses.size() + newly_connected.size());

        // add new nodes that are directly connected
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

    initializeOrientation(graph, _local_poses);

    spdlog::info("Queueing {} image nodes, {} edges for graph relaxation", _local_poses.size(),
                 _edges_to_optimize.size());
}

void RelaxGroup::build_optimization_edges(const MeasurementGraph &graph,
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
            _directly_connected.insert(edge->getDest());
            if (_nodes_to_optimize.find(edge->getDest()) != _nodes_to_optimize.end())
            {
                _edges_to_optimize.insert(edge_id);
            }
        }
        else if (edge->getDest() == node_id &&
                 ideally_connected_nodes.find(edge->getSource()) != ideally_connected_nodes.end())
        {
            _directly_connected.insert(edge->getSource());
            if (_nodes_to_optimize.find(edge->getSource()) != _nodes_to_optimize.end())
            {
                _edges_to_optimize.insert(edge_id);
            }
        }
    }
}

void RelaxGroup::run(const MeasurementGraph &graph)
{
    if (_local_poses.size() > 0)
    {
        switch (_relax_type)
        {
        case RelaxType::MEASUREMENT_RELAX_POINTS: {
            PerformanceMeasure p("Relax runner points");
            relax3dPointMeasurements(graph, _local_poses, _edges_to_optimize);
            break;
        }
        case RelaxType::MEASUREMENT_RELAX_PLANE: {
            PerformanceMeasure p("Relax runner plane");
            relaxGroundPlaneMeasurements(graph, _local_poses, _edges_to_optimize);
            break;
        }
        case RelaxType::RELATIVE_RELAX: {
            PerformanceMeasure p("Relax runner relative");
            relaxDecompositions(graph, _local_poses, _edges_to_optimize);
            break;
        }
        }
    }
}

std::vector<size_t> RelaxGroup::finalize(MeasurementGraph &graph)
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
