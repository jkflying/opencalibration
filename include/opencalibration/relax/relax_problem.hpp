#pragma once

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geometry/intersection.hpp>
#include <opencalibration/relax/grid_filter.hpp>
#include <opencalibration/types/feature_track.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/plane.hpp>
#include <opencalibration/types/point_cloud.hpp>
#include <opencalibration/types/relax_options.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <spdlog/spdlog.h>

#include <ankerl/unordered_dense.h>

namespace opencalibration
{

struct OptimizationPackage
{
    const camera_relations *relations;

    struct PoseOpt
    {
        Eigen::Vector3d *loc_ptr = nullptr;
        Eigen::Quaterniond *rot_ptr = nullptr;
        CameraModel *model_ptr = nullptr;
        bool optimize = true;
        size_t node_id = 0;
    } source, dest;
};

class RelaxProblem
{
  public:
    RelaxProblem();
    void setupDecompositionProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                   const ankerl::unordered_dense::set<size_t> &edges_to_optimize);

    void setupGroundPlaneProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                 ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                                 const ankerl::unordered_dense::set<size_t> &edges_to_optimize, const RelaxOptionSet &options);

    void setupGroundMeshProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                                const ankerl::unordered_dense::set<size_t> &edges_to_optimize, const RelaxOptionSet &options,
                                const std::vector<surface_model> &previousSurfaces);

    void setup3dPointProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                             ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                             const ankerl::unordered_dense::set<size_t> &edges_to_optimize, const RelaxOptionSet &options);

    void relaxObservedModelOnly(); // only 3d points and ground plane
    void solve();

    surface_model getSurfaceModel();

  protected:
    void initialize(std::vector<NodePose> &nodes, ankerl::unordered_dense::map<size_t, CameraModel> &cam_models);
    bool shouldAddEdgeToOptimization(const ankerl::unordered_dense::set<size_t> &edges_to_optimize, size_t edge_id);

    OptimizationPackage::PoseOpt nodeid2poseopt(const MeasurementGraph &graph, size_t node_id,
                                                bool load_cam_model = true);

    void addRelationCost(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge);

    void gridFilterMatchesPerImage(const MeasurementGraph &graph, const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
                                   double grid_cell_image_fraction);

    void addPointMeasurementsCost(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge,
                                  const RelaxOptionSet &options);

    void addRayTriangleMeasurementCost(const MeasurementGraph &graph, size_t edge_id,
                                       const MeasurementGraph::Edge &edge, const RelaxOptionSet &options);

    void initializeGroundPlane();
    void initializeGroundMesh(const std::vector<surface_model> &previousSurfaces, bool useMinimalMesh = false);

    void addDownwardsPrior();
    void addMeshFlatPrior();

    void trackRadialObservation(double *radial_data, size_t pixels_rows, size_t pixels_cols, double focal_length);
    void addMonotonicityCosts();

    ceres::Solver::Options _solver_options;
    ceres::LossFunctionWrapper _loss;

    ceres::Problem::Options _problemOptions;
    ceres::EigenQuaternionManifold _quat_parameterization;

    ceres::SubsetManifold _brown2_parameterization;
    ceres::SubsetManifold _brown24_parameterization;
    ceres::EuclideanManifold<3> _brown246_parameterization;

    std::unique_ptr<ceres::Problem> _problem;

    ankerl::unordered_dense::map<size_t, ankerl::unordered_dense::map<size_t, GridFilter<const feature_match_denormalized *>>> _grid_filter;

    ceres::Solver::Summary _summary;
    ceres::Solver _solver;

    ankerl::unordered_dense::map<size_t, NodePose *> _nodes_to_optimize;
    ankerl::unordered_dense::map<size_t, CameraModel *> _cam_models_to_optimize;
    ankerl::unordered_dense::map<size_t, InverseDifferentiableCameraModel<double>> _inverse_cam_model_to_optimize;
    ankerl::unordered_dense::set<size_t> _edges_used;

    // Radial distortion monotonicity tracking
    struct MonotonicityInfo
    {
        size_t observation_count = 0;
        double r_max = 0;
    };
    ankerl::unordered_dense::map<double *, MonotonicityInfo> _radial_monotonicity_info;

    // Surface models
    using track_vec = std::vector<FeatureTrack, Eigen::aligned_allocator<FeatureTrack>>;
    ankerl::unordered_dense::map<size_t, track_vec> _edge_tracks;
    MeshGraph _mesh;
    std::vector<double> _mesh_initial_z;
};

} // namespace opencalibration
