#include <opencalibration/relax/relax.hpp>

#include <opencalibration/relax/relax_problem.hpp>

namespace opencalibration
{

void initializeOrientation(const MeasurementGraph &graph, std::vector<NodePose> &nodes)
{
    (void)graph;
    static const Eigen::Quaterniond DOWN_ORIENTED_NORTH(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    for (NodePose &node_pose : nodes)
    {
        if (!node_pose.orientation.coeffs().allFinite())
        {
            node_pose.orientation = DOWN_ORIENTED_NORTH;
        }
    }
}

void relaxDecompositions(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                         const std::unordered_set<size_t> &edges_to_optimize)
{
    RelaxProblem rp;
    rp.initialize(nodes);
    rp.solverOptions.initial_trust_region_radius = 0.1;

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && rp.shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            rp.addRelationCost(graph, edge_id, *edge);
        }
    }

    rp.addDownwardsPrior();

    rp.solve();
}

void relax3dPointMeasurements(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                              const std::unordered_set<size_t> &edges_to_optimize)
{
    RelaxProblem rp;
    rp.initialize(nodes);
    rp.huber_loss.reset(new ceres::HuberLoss(10));

    rp.gridFilterMatchesPerImage(graph, edges_to_optimize);

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && rp.shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            rp.addPointMeasurementsCost(graph, edge_id, *edge);
        }
    }

    rp.solverOptions.max_num_iterations = 300;
    rp.solverOptions.linear_solver_type = ceres::SPARSE_SCHUR;

    rp.solve();
}

void relaxGroundPlaneMeasurements(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                  const std::unordered_set<size_t> &edges_to_optimize)
{
    RelaxProblem rp;
    rp.initialize(nodes);
    rp.initializeGroundPlane();
    rp.huber_loss.reset(new ceres::HuberLoss(1 * M_PI / 180));
    rp.gridFilterMatchesPerImage(graph, edges_to_optimize);

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && rp.shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            rp.addGlobalPlaneMeasurementsCost(graph, edge_id, *edge);
        }
    }

    rp.solve();
}

} // namespace opencalibration
