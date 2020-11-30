#include <opencalibration/relax/relax.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>

#include <unordered_set>

namespace opencalibration
{

static const Eigen::Quaterniond DOWN_ORIENTED_NORTH = Eigen::Quaterniond::Identity(); // TODO FIXME

void initializeOrientation(const std::vector<size_t> &node_ids, MeasurementGraph &graph)
{
    std::vector<std::pair<double, Eigen::Quaterniond>> hypotheses;
    for (size_t node_id : node_ids)
    {
        MeasurementGraph::Node *node = graph.getNode(node_id);
        if (node == nullptr)
            return;

        // get connections
        hypotheses.clear();
        hypotheses.reserve(node->getEdges().size());

        for (size_t edge_id : node->getEdges())
        {
            MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
            size_t other_node_id = edge->getDest() == node_id ? edge->getSource() : edge->getDest();
            MeasurementGraph::Node *other_node = graph.getNode(other_node_id);

            Eigen::Quaterniond transform = edge->getDest() == node_id ? edge->payload.relative_rotation.inverse()
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

        node->payload.orientation.coeffs() = weight_sum > 0 ? vec_sum / weight_sum : DOWN_ORIENTED_NORTH.coeffs();
    }
}

void relaxExternal(const MeasurementGraph &graph, std::vector<NodePose> &nodes)
{
    (void)graph;
    (void)nodes;

    ceres::Problem::Options problemOptions;
    problemOptions.cost_function_ownership = ceres::TAKE_OWNERSHIP;
    problemOptions.loss_function_ownership = ceres::TAKE_OWNERSHIP;
    problemOptions.local_parameterization_ownership = ceres::TAKE_OWNERSHIP;

    ceres::Problem problem(problemOptions);

    ceres::Solver::Options solverOptions;
    solverOptions.num_threads = 1;
    solverOptions.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;

    ceres::Solver solver;
    solver.Solve(solverOptions, &problem, &summary);
}
} // namespace opencalibration
