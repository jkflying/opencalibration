#include <opencalibration/relax/relax.hpp>

#include <opencalibration/performance/performance.hpp>
#include <opencalibration/relax/relax_problem.hpp>

namespace
{

using namespace opencalibration;

static const Eigen::Quaterniond DOWN_ORIENTED_NORTH(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));

struct Backend
{
    using run_f =
        std::function<void(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                           std::unordered_map<size_t, CameraModel> &cam_models,
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
                             std::unordered_map<size_t, CameraModel> &cam_models,
                             const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &) {
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
                          });
    backends.emplace_back(RelaxOptionSet{Option::LENS_DISTORTIONS_TANGENTIAL, Option::LENS_DISTORTIONS_RADIAL,
                                         Option::LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION,
                                         Option::LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION,
                                         Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION,
                                         Option::FOCAL_LENGTH, Option::ORIENTATION, Option::POINTS_3D},
                          [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                             std::unordered_map<size_t, CameraModel> &cam_models,
                             const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options) {
                              PerformanceMeasure p("Relax runner 3d points frt");
                              RelaxProblem rp;
                              rp.setup3dPointProblem(graph, nodes, cam_models, edges_to_optimize, options);
                              rp.solve();
                          });
    backends.emplace_back(RelaxOptionSet{Option::LENS_DISTORTIONS_RADIAL,
                                         Option::LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION,
                                         Option::LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION,
                                         Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION,
                                         Option::FOCAL_LENGTH, Option::ORIENTATION, Option::POINTS_3D},
                          [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                             std::unordered_map<size_t, CameraModel> &cam_models,
                             const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options) {
                              PerformanceMeasure p("Relax runner 3d points fr");
                              RelaxProblem rp;
                              rp.setup3dPointProblem(graph, nodes, cam_models, edges_to_optimize, options);
                              rp.solve();
                          });
    backends.emplace_back(RelaxOptionSet{Option::FOCAL_LENGTH, Option::ORIENTATION, Option::POINTS_3D},
                          [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                             std::unordered_map<size_t, CameraModel> &cam_models,
                             const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options) {
                              PerformanceMeasure p("Relax runner 3d points f");
                              RelaxProblem rp;
                              rp.setup3dPointProblem(graph, nodes, cam_models, edges_to_optimize, options);
                              rp.solve();
                          });
    backends.emplace_back(RelaxOptionSet{Option::ORIENTATION, Option::POINTS_3D},
                          [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                             std::unordered_map<size_t, CameraModel> &cam_models,
                             const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options) {
                              PerformanceMeasure p("Relax runner 3d points");
                              RelaxProblem rp;
                              rp.setup3dPointProblem(graph, nodes, cam_models, edges_to_optimize, options);
                              rp.solve();
                          });
    backends.emplace_back(RelaxOptionSet{Option::ORIENTATION, Option::GROUND_PLANE},
                          [](const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                             std::unordered_map<size_t, CameraModel> &cam_models,
                             const std::unordered_set<size_t> &edges_to_optimize, const RelaxOptionSet &options) {
                              PerformanceMeasure p("Relax runner ground plane");
                              RelaxProblem rp;
                              rp.setupGroundPlaneProblem(graph, nodes, cam_models, edges_to_optimize, options);
                              rp.relaxObservedModelOnly();
                              rp.solve();
                          });
    return backends;
}

static const std::vector<Backend> backends = getBackends();
} // namespace

namespace opencalibration
{

void relax(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
           std::unordered_map<size_t, CameraModel> &cam_models, const std::unordered_set<size_t> &edges_to_optimize,
           const RelaxOptionSet &options)
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
        backends[best_index].runner(graph, nodes, cam_models, edges_to_optimize, options);
    }
}
} // namespace opencalibration
