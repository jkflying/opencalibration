#include <opencalibration/relax/relax.hpp>

#include <opencalibration/performance/performance.hpp>
#include <opencalibration/relax/relax_problem.hpp>
#include <opencalibration/types/surface_model.hpp>

namespace
{

using namespace opencalibration;

static const Eigen::Quaterniond DOWN_ORIENTED_NORTH(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));

surface_model runRelativeOrientation(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                     ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                                     const ankerl::unordered_dense::set<size_t> &edges_to_optimize)
{
    (void)cam_models;
    PerformanceMeasure p("Relax runner relative");

    for (auto &node : nodes)
    {
        if (node.orientation.coeffs().hasNaN())
        {
            node.orientation = DOWN_ORIENTED_NORTH;

            // add just one image at a time with a bad initial orientation, then force an
            // optimize otherwise we could disturb the other images and end up in a bad local
            // minima the setup will already discard images which are uninitialized
            RelaxProblem rp;
            rp.setupDecompositionProblem(graph, nodes, edges_to_optimize);
            rp.solve();
        }
    }

    // finally do an optimize for the whole batch
    RelaxProblem rp;
    rp.setupDecompositionProblem(graph, nodes, edges_to_optimize);
    rp.solve();

    return rp.getSurfaceModel();
}

surface_model runGroundPlane(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                             ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                             const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
                             const RelaxOptionSet &options)
{
    PerformanceMeasure p("Relax runner ground plane");

    Eigen::Quaterniond previous_node_orientation = DOWN_ORIENTED_NORTH;
    for (auto &node : nodes)
    {
        if (node.orientation.coeffs().hasNaN())
        {
            node.orientation = previous_node_orientation;

            // add just one image at a time with a bad initial orientation, and optimize it while holding others
            // still if there are enough images to do so
            // The setup will already discard images which are uninitialized
            if (graph.size_nodes() > 2 * nodes.size())
            {
                std::vector<NodePose> justThis{node};
                RelaxProblem rp;
                rp.setupGroundPlaneProblem(graph, justThis, cam_models, edges_to_optimize, options);
                rp.relaxObservedModelOnly();
                rp.solve();
                node = justThis[0];
            }
            else
            {
                RelaxProblem rp;
                rp.setupGroundPlaneProblem(graph, nodes, cam_models, edges_to_optimize, options);
                rp.relaxObservedModelOnly();
                rp.solve();
            }
        }
        previous_node_orientation = node.orientation;
    }

    RelaxProblem rp;
    rp.setupGroundPlaneProblem(graph, nodes, cam_models, edges_to_optimize, options);
    rp.relaxObservedModelOnly();
    rp.solve();

    return rp.getSurfaceModel();
}

surface_model runGroundMesh(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                            ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                            const ankerl::unordered_dense::set<size_t> &edges_to_optimize, const RelaxConfig &config,
                            const std::vector<surface_model> &previousSurfaces)
{
    PerformanceMeasure p("Relax runner ground mesh");
    RelaxProblem rp;
    rp.setupGroundMeshProblem(graph, nodes, cam_models, edges_to_optimize, config.options, previousSurfaces,
                              config.ground_mesh_grid_fraction);
    rp.relaxObservedModelOnly();
    rp.solve();

    return rp.getSurfaceModel();
}

surface_model runPoints(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                        ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                        const ankerl::unordered_dense::set<size_t> &edges_to_optimize, const RelaxOptionSet &options)
{
    PerformanceMeasure p("Relax runner 3d points");
    RelaxProblem rp;
    rp.setup3dPointProblem(graph, nodes, cam_models, edges_to_optimize, options);
    rp.relaxObservedModelOnly();
    rp.solve();

    return rp.getSurfaceModel();
}

} // namespace

namespace opencalibration
{

surface_model relax(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                    ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                    const ankerl::unordered_dense::set<size_t> &edges_to_optimize, const RelaxConfig &config,
                    const std::vector<surface_model> &previousSurfaces)
{
    if (config.options.get(Option::GROUND_MESH))
        return runGroundMesh(graph, nodes, cam_models, edges_to_optimize, config, previousSurfaces);
    if (config.options.get(Option::POINTS_3D))
        return runPoints(graph, nodes, cam_models, edges_to_optimize, config.options);
    if (config.options.get(Option::GROUND_PLANE))
        return runGroundPlane(graph, nodes, cam_models, edges_to_optimize, config.options);
    return runRelativeOrientation(graph, nodes, cam_models, edges_to_optimize);
}

} // namespace opencalibration
