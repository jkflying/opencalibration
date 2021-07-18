#include <opencalibration/relax/relax.hpp>

#include <opencalibration/performance/performance.hpp>
#include <opencalibration/relax/relax_problem.hpp>

namespace
{

using namespace opencalibration;

struct Backend
{
    using run_f =
        std::function<void(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                           const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options)>;
    Backend(const RelaxOptionSet &caps, run_f &&func) : capabilities(caps), runner(func)
    {
    }
    RelaxOptionSet capabilities;
    run_f runner;
};

std::vector<Backend> getBackends()
{
    std::vector<Backend> backends;
    backends.emplace_back(RelaxOptionSet{Option::ORIENTATION},
                          [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                             const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &) {
                              PerformanceMeasure p("Relax runner relative");
                              RelaxProblem rp;
                              rp.setupDecompositionProblem(graph, nodes, edges_to_optimize);
                              rp.solve();
                          });
    backends.emplace_back(RelaxOptionSet{Option::ORIENTATION, Option::POINTS_3D},
                          [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                             const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options) {
                              PerformanceMeasure p("Relax runner 3d points");
                              RelaxProblem rp;
                              rp.setup3dPointProblem(graph, nodes, edges_to_optimize, options);
                              rp.solve();
                          });
    backends.emplace_back(RelaxOptionSet{Option::ORIENTATION, Option::GROUND_PLANE},
                          [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                             const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options) {
                              PerformanceMeasure p("Relax runner ground plane");
                              RelaxProblem rp;
                              rp.setupGroundPlaneProblem(graph, nodes, edges_to_optimize, options);
                              rp.relaxObservedModelOnly();
                              rp.solve();
                          });
    return backends;
}

std::vector<Backend> backends = getBackends();
} // namespace

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

void relax(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
           const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options)
{

    size_t best_index = std::numeric_limits<size_t>::max();
    int32_t best_index_excess = std::numeric_limits<int32_t>::max();

    for (size_t i = 0; i < backends.size(); i++)
    {
        auto &backend = backends[i];

        if (backend.capabilities.hasAll(options))
        {
            int32_t excess = backend.capabilities.excess(options);
            if (excess < best_index_excess)
            {
                best_index = i;
                best_index_excess = excess;
            }
        }
    }

    if (best_index != std::numeric_limits<size_t>::max())
    {
        backends[best_index].runner(graph, nodes, edges_to_optimize, options);
    }
}
} // namespace opencalibration
