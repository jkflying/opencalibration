#include <opencalibration/relax/relax_group.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/model_inliers/ransac.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/relax/relax.hpp>
#include <opencalibration/types/correspondence.hpp>

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
                      const RelaxOptionSet &relax_options)
{
    _directly_connected.clear();
    _edges_to_optimize.clear();
    _nodes_to_optimize.clear();
    _local_poses.clear();

    _relax_options = relax_options;
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
        _camera_models[node->payload.model->id] = *node->payload.model;
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
            _camera_models[node->payload.model->id] = *node->payload.model;
            build_optimization_edges(graph, imageGPSLocations, pose.node_id);
        }
    }

    std::sort(_local_poses.begin(), _local_poses.end(), [&graph](const NodePose &a, const NodePose &b) {
        const std::string &a_s = graph.getNode(a.node_id)->payload.path, &b_s = graph.getNode(b.node_id)->payload.path;
        return a_s < b_s;
    });

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

surface_model opencalibration::RelaxGroup::run(const MeasurementGraph &graph)
{
    return relax(graph, _local_poses, _camera_models, _edges_to_optimize, _relax_options);
}

std::vector<size_t> RelaxGroup::finalize(MeasurementGraph &graph)
{
    std::vector<size_t> optimized_ids;
    optimized_ids.reserve(_local_poses.size());
    bool model_changed = _relax_options.hasAny({Option::FOCAL_LENGTH, Option::PRINCIPAL_POINT,
                                                Option::LENS_DISTORTIONS_RADIAL, Option::LENS_DISTORTIONS_TANGENTIAL});
    for (const auto &pose : _local_poses)
    {
        auto *node = graph.getNode(pose.node_id);
        node->payload.orientation = pose.orientation;
        node->payload.position = pose.position;
        if (model_changed && !(*node->payload.model == _camera_models[node->payload.model->id]))
        {
            *node->payload.model = _camera_models[node->payload.model->id];
        }
        optimized_ids.push_back(pose.node_id);
    }

    if (model_changed)
    {
        // recalculate all of the edge inliers based on camera models changing. Just iterate from current inliers,
        // recalculate model.
        for (auto eiter = graph.edgebegin(); eiter != graph.edgeend(); ++eiter)
        {
            auto &edge = eiter->second;
            const auto *source = graph.getNode(edge.getSource());
            const auto *dest = graph.getNode(edge.getDest());

            const std::vector<correspondence> correspondences =
                distort_keypoints(source->payload.features, dest->payload.features, edge.payload.matches,
                                  *source->payload.model, *dest->payload.model);

            std::vector<bool> inliers(correspondences.size(), false);

            for (const auto &old_inlier : edge.payload.inlier_matches)
            {
                inliers[old_inlier.match_index] = true;
            }

            // do a sort of 'maximum likelihood' based on the previous inliers
            homography_model h;
            for (int i = 0; i < 3; i++)
            {
                h.fitInliers(correspondences, inliers);
                h.evaluate(correspondences, inliers);
            }
            edge.payload.ransac_relation = h.homography;
            edge.payload.relationType = camera_relations::RelationType::HOMOGRAPHY;

            bool can_decompose = h.decompose(correspondences, inliers, edge.payload.relative_poses);
            size_t num_inliers = std::count(inliers.begin(), inliers.end(), true);

            edge.payload.inlier_matches.clear();
            if (can_decompose && num_inliers > h.MINIMUM_POINTS * 1.5)
            {
                assembleInliers(edge.payload.matches, inliers, source->payload.features, dest->payload.features,
                                edge.payload.inlier_matches);
            }
        }
    }

    _local_poses.clear();
    return optimized_ids;
}

} // namespace opencalibration
