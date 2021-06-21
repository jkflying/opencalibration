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
    rp.setupDecompositionProblem(graph, nodes, edges_to_optimize);
    rp.solve();
}

void relax3dPointMeasurements(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                              const std::unordered_set<size_t> &edges_to_optimize)
{
    RelaxProblem rp;
    rp.setup3dPointProblem(graph, nodes, edges_to_optimize);
    rp.solve();
}

void relaxGroundPlaneMeasurements(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                  const std::unordered_set<size_t> &edges_to_optimize)
{
    RelaxProblem rp;
    rp.setupGroundPlaneProblem(graph, nodes, edges_to_optimize);
    rp.relaxObservedModelOnly();
    rp.solve();
}

} // namespace opencalibration
