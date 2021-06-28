#pragma once

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geometry/intersection.hpp>
#include <opencalibration/relax/grid_filter.hpp>
#include <opencalibration/types/feature_track.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/plane.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <ceres/local_parameterization.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <spdlog/spdlog.h>

#include <unordered_map>
#include <unordered_set>

namespace opencalibration
{

struct OptimizationPackage
{
    const camera_relations *relations;

    struct PoseOpt
    {
        Eigen::Vector3d *loc_ptr = nullptr;
        Eigen::Quaterniond *rot_ptr = nullptr;
        bool optimize = true;
        size_t node_id = 0;
    } source, dest;
};

class RelaxProblem
{
  public:
    RelaxProblem();
    void setupDecompositionProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                   const std::unordered_set<size_t> &edges_to_optimize);

    void setupGroundPlaneProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                 const std::unordered_set<size_t> &edges_to_optimize);

    void setup3dPointProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                             const std::unordered_set<size_t> &edges_to_optimize);

    void setup3dTracksProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                              const std::unordered_set<size_t> &edges_to_optimize);

    void relaxObservedModelOnly(); // only 3d points and ground plane
    void solve();

  protected:
    void initialize(std::vector<NodePose> &nodes);
    bool shouldAddEdgeToOptimization(const std::unordered_set<size_t> &edges_to_optimize, size_t edge_id);

    OptimizationPackage::PoseOpt nodeid2poseopt(const MeasurementGraph &graph, size_t node_id);

    void addRelationCost(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge);

    void gridFilterMatchesPerImage(const MeasurementGraph &graph, const std::unordered_set<size_t> &edges_to_optimize);

    void insertEdgeTracks(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge);
    void filterTracks(const MeasurementGraph &graph);
    void compileEdgeTracks(const MeasurementGraph &graph);
    void addTrackCosts(const MeasurementGraph &graph);
    void addPointMeasurementsCost(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge);

    void addGlobalPlaneMeasurementsCost(const MeasurementGraph &graph, size_t edge_id,
                                        const MeasurementGraph::Edge &edge);

    void initializeGroundPlane();

    void addDownwardsPrior();

    void *_log_forwarder_dependency;

    ceres::Solver::Options _solver_options;
    ceres::LossFunctionWrapper _loss;

    ceres::Problem::Options _problemOptions;
    ceres::EigenQuaternionParameterization _quat_parameterization;
    std::unique_ptr<ceres::Problem> _problem;

    std::unordered_map<size_t, GridFilter<const feature_match_denormalized *>> _grid_filter;

    ceres::Solver::Summary _summary;
    ceres::Solver _solver;

    std::unordered_map<size_t, NodePose *> _nodes_to_optimize;
    std::unordered_set<size_t> _edges_used;

    // Surface models
    using track_vec = std::vector<FeatureTrack, Eigen::aligned_allocator<FeatureTrack>>;
    plane_3_corners<double> _global_plane;
    std::unordered_map<size_t, track_vec> _edge_tracks;
    track_vec _tracks;

    // temporary for building tracks
    std::unordered_map<NodeIdFeatureIndex, std::vector<NodeIdFeatureIndex>> _node_id_feature_index_tracklinks;
};

} // namespace opencalibration
