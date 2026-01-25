#include <opencalibration/relax/relax.hpp>

#include <opencalibration/performance/performance.hpp>
#include <opencalibration/relax/relax_problem.hpp>
#include <opencalibration/surface/refine_mesh.hpp>
#include <opencalibration/types/surface_model.hpp>

namespace
{

using namespace opencalibration;

static const Eigen::Quaterniond DOWN_ORIENTED_NORTH(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));

struct Backend
{
    using run_f = std::function<surface_model(
        const MeasurementGraph &graph, std::vector<NodePose> &nodes,
        std::unordered_map<size_t, CameraModel> &cam_models, const std::unordered_set<size_t> &edges_to_optimize,
        const RelaxOptionSet &options, const std::vector<surface_model> &previousSurfaces)>;
    Backend(const RelaxOptionSet &caps, run_f &&func) : capabilities(caps), runner(func)
    {
    }
    RelaxOptionSet capabilities;
    run_f runner;
};

std::vector<Backend> getBackends()
{
    std::vector<Backend> backends;
    auto points_solver = [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                            std::unordered_map<size_t, CameraModel> &cam_models,
                            const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options,
                            const std::vector<surface_model> &) -> surface_model {
        PerformanceMeasure p("Relax runner 3d points");
        RelaxProblem rp;
        rp.setup3dPointProblem(graph, nodes, cam_models, edges_to_optimize, options);
        rp.relaxObservedModelOnly();
        rp.solve();

        return rp.getSurfaceModel();
    };

    auto mesh_solver = [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                          std::unordered_map<size_t, CameraModel> &cam_models,
                          const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options,
                          const std::vector<surface_model> &previousSurfaces) -> surface_model {
        PerformanceMeasure p("Relax runner ground mesh");
        RelaxProblem rp;
        rp.setupGroundMeshProblem(graph, nodes, cam_models, edges_to_optimize, options, previousSurfaces);
        rp.relaxObservedModelOnly();
        rp.solve();

        return rp.getSurfaceModel();
    };

    auto ground_plane_solver = [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                  std::unordered_map<size_t, CameraModel> &cam_models,
                                  const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options,
                                  const std::vector<surface_model> &) -> surface_model {
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
    };

    auto relative_orientation_solver = [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                          std::unordered_map<size_t, CameraModel> &cam_models,
                                          const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &,
                                          const std::vector<surface_model> &) -> surface_model {
        // doesn't use the camera model at all, just decomposed relative orientations
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
    };

    backends.emplace_back(RelaxOptionSet{Option::ORIENTATION}, relative_orientation_solver);

    backends.emplace_back(RelaxOptionSet{Option::ORIENTATION, Option::GROUND_PLANE}, ground_plane_solver);

    backends.emplace_back(
        RelaxOptionSet{Option::LENS_DISTORTIONS_RADIAL, Option::LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION,
                       Option::LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION,
                       Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION, Option::FOCAL_LENGTH,
                       Option::PRINCIPAL_POINT, Option::ORIENTATION, Option::GROUND_MESH},
        mesh_solver);

    // Adaptive mesh solver: starts with minimal mesh, iteratively refines and optimizes
    auto adaptive_mesh_solver = [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                   std::unordered_map<size_t, CameraModel> &cam_models,
                                   const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options,
                                   const std::vector<surface_model> &previousSurfaces) -> surface_model {
        PerformanceMeasure p("Relax runner adaptive mesh");

        const size_t maxPointsPerTriangle = 20;
        const int maxIterations = 20;

        // Start with minimal mesh option enabled
        RelaxOptionSet minimalOptions = options;
        minimalOptions.set(Option::MINIMAL_MESH, true);

        surface_model currentSurface;
        std::vector<surface_model> surfaces = previousSurfaces;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Setup and solve optimization problem
            RelaxProblem rp;
            rp.setupGroundMeshProblem(graph, nodes, cam_models, edges_to_optimize, iter == 0 ? minimalOptions : options,
                                      surfaces);
            rp.relaxObservedModelOnly();
            rp.solve();

            currentSurface = rp.getSurfaceModel();

            // Count points per triangle
            size_t maxPoints = 0;
            size_t trianglesAboveThreshold = 0;
            auto counts = countPointsPerTriangle(currentSurface.mesh, currentSurface.cloud);
            for (const auto &[key, count] : counts)
            {
                maxPoints = std::max(maxPoints, count);
                if (count > maxPointsPerTriangle)
                {
                    trianglesAboveThreshold++;
                }
            }

            spdlog::info("Adaptive mesh iteration {}: {} nodes, {} edges, max {} points/triangle, {} triangles above "
                         "threshold",
                         iter, currentSurface.mesh.size_nodes(), currentSurface.mesh.size_edges(), maxPoints,
                         trianglesAboveThreshold);

            // Check convergence
            if (trianglesAboveThreshold == 0)
            {
                spdlog::info("Adaptive mesh converged: all triangles have <= {} points", maxPointsPerTriangle);
                break;
            }

            // Refine triangles with too many points
            size_t created = refineByPointDensity(currentSurface.mesh, currentSurface.cloud, maxPointsPerTriangle, 1);
            if (created == 0)
            {
                spdlog::info("Adaptive mesh: no refinement possible, stopping");
                break;
            }

            spdlog::info("Adaptive mesh: refined {} triangles", created);

            // Use current surface as previous for next iteration
            surfaces = {currentSurface};
        }

        return currentSurface;
    };

    backends.emplace_back(RelaxOptionSet{Option::LENS_DISTORTIONS_RADIAL,
                                         Option::LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION,
                                         Option::LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION,
                                         Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION,
                                         Option::FOCAL_LENGTH, Option::PRINCIPAL_POINT, Option::ORIENTATION,
                                         Option::GROUND_MESH, Option::MINIMAL_MESH, Option::ADAPTIVE_REFINE},
                          adaptive_mesh_solver);

    backends.emplace_back(RelaxOptionSet{Option::LENS_DISTORTIONS_TANGENTIAL, Option::LENS_DISTORTIONS_RADIAL,
                                         Option::LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION,
                                         Option::LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION,
                                         Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION,
                                         Option::FOCAL_LENGTH, Option::PRINCIPAL_POINT, Option::ORIENTATION,
                                         Option::POINTS_3D},
                          points_solver);
    return backends;
}

static const std::vector<Backend> backends = getBackends();
} // namespace

namespace opencalibration
{

surface_model relax(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                    std::unordered_map<size_t, CameraModel> &cam_models,
                    const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options,
                    const std::vector<surface_model> &previousSurfaces)
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
        return backends[best_index].runner(graph, nodes, cam_models, edges_to_optimize, options, previousSurfaces);
    }

    return surface_model();
}
} // namespace opencalibration
