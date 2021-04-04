#include <opencalibration/relax/relax.hpp>

#include <opencalibration/relax/relax_problem.hpp>

namespace opencalibration
{

void initializeOrientation(const MeasurementGraph &graph, std::vector<NodePose> &nodes)
{
    static const Eigen::Quaterniond DOWN_ORIENTED_NORTH(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    std::vector<std::pair<double, Eigen::Quaterniond>> hypotheses;
    for (NodePose &node_pose : nodes)
    {
        const MeasurementGraph::Node *node = graph.getNode(node_pose.node_id);
        if (node == nullptr)
        {
            spdlog::error("Null node referenced from optimization list");
            continue;
        }

        // get connections
        hypotheses.clear();
        hypotheses.reserve(node->getEdges().size());

        for (size_t edge_id : node->getEdges())
        {
            const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
            size_t other_node_id = edge->getDest() == node_pose.node_id ? edge->getSource() : edge->getDest();
            const MeasurementGraph::Node *other_node = graph.getNode(other_node_id);

            Eigen::Quaterniond transform = edge->getDest() == node_pose.node_id
                                               ? edge->payload.relative_rotation.inverse()
                                               : edge->payload.relative_rotation;

            // TODO: use relative positions as well when NaN
            bool other_ori_nan = other_node->payload.orientation.coeffs().hasNaN();
            Eigen::Quaterniond other_orientation =
                other_ori_nan ? DOWN_ORIENTED_NORTH : other_node->payload.orientation;
            double weight = other_ori_nan ? 0.1 : 1;
            Eigen::Quaterniond hypothesis = transform * other_orientation;

            hypotheses.emplace_back(weight, hypothesis);
        }

        // for now just make a dumb weighted average
        double weight_sum = 0;
        Eigen::Vector4d vec_sum;
        for (const auto &h : hypotheses)
        {
            weight_sum += h.first;
            vec_sum += h.first * h.second.coeffs();
        }

        node_pose.orientation.coeffs() = weight_sum > 0 ? vec_sum : DOWN_ORIENTED_NORTH.coeffs();
        node_pose.orientation.normalize();
    }
}

void relaxDecompositions(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                         const std::unordered_set<size_t> &edges_to_optimize)
{
    RelaxProblem rp;
    rp.initialize(nodes);

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && rp.shouldOptimizeEdge(edges_to_optimize, edge_id, *edge))
        {
            rp.addRelationCost(graph, edge_id, *edge);
        }
    }

    rp.addDownwardsPrior(nodes);
    rp.setConstantBlocks(graph);

    rp.solve(nodes);
}

void relaxMeasurements(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                       const std::unordered_set<size_t> &edges_to_optimize)
{
    RelaxProblem rp;
    rp.initialize(nodes);
    rp.huber_loss.reset(new ceres::HuberLoss(10));

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && rp.shouldOptimizeEdge(edges_to_optimize, edge_id, *edge))
        {
            rp.addMeasurementsCost(graph, edge_id, *edge);
        }
    }

    // rp.addDownwardsPrior(nodes); the downwards prior hopefully isn't necessary at this stage in the bundle
    rp.setConstantBlocks(graph);

    rp.solverOptions.max_num_iterations = 300;
    rp.solverOptions.linear_solver_type = ceres::SPARSE_SCHUR;

    rp.solve(nodes);
}
} // namespace opencalibration
