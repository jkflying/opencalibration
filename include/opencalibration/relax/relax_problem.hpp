#pragma once

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geometry/intersection.hpp>
#include <opencalibration/relax/grid_filter.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/plane.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>

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

struct TrackLink
{
    size_t edge_id, denormalized_match_index;

    int operator<(const TrackLink &other)
    {
        return std::make_pair(edge_id, denormalized_match_index) <
               std::make_pair(other.edge_id, other.denormalized_match_index);
    }

    int operator==(const TrackLink &other)
    {
        return edge_id == other.edge_id && denormalized_match_index == other.denormalized_match_index;
    }
};

struct FeatureTrack
{
    Eigen::Vector3d point{NAN, NAN, NAN};
    std::vector<TrackLink> measurements;
};

class RelaxProblem
{
  public:
    RelaxProblem();

    void initialize(std::vector<NodePose> &nodes);
    bool shouldAddEdgeToOptimization(const std::unordered_set<size_t> &edges_to_optimize, size_t edge_id);

    OptimizationPackage::PoseOpt nodeid2poseopt(const MeasurementGraph &graph, size_t node_id);

    void addRelationCost(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge);

    void gridFilterMatchesPerImage(const MeasurementGraph &graph, const std::unordered_set<size_t> &edges_to_optimize);

    void insertEdgeTracks(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge);
    void compileEdgeTracks(const MeasurementGraph &graph);
    void addPointMeasurementsCost(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge);

    void addGlobalPlaneMeasurementsCost(const MeasurementGraph &graph, size_t edge_id,
                                        const MeasurementGraph::Edge &edge);

    void initializeGroundPlane();

    void addDownwardsPrior();

    void solve();

    ceres::Solver::Options solverOptions;
    std::unique_ptr<ceres::HuberLoss> huber_loss;

  private:
    ceres::Problem::Options _problemOptions;
    ceres::EigenQuaternionParameterization _quat_parameterization;
    std::unique_ptr<ceres::Problem> _problem;

    std::unordered_map<size_t, GridFilter<const feature_match_denormalized *>> _grid_filter;

    ceres::Solver::Summary _summary;
    ceres::Solver _solver;

    std::unordered_map<size_t, NodePose *> _nodes_to_optimize;
    std::unordered_set<size_t> _edges_used;

    // Surface models
    plane_3_corners<double> _global_plane;
    std::unordered_map<size_t, point_cloud> _edge_points;
    std::vector<FeatureTrack, Eigen::aligned_allocator<FeatureTrack>> _tracks;

    // temporary for building tracks
    struct NodeIdFeatureIndex
    {
        size_t node_id, feature_index;
        bool operator==(const NodeIdFeatureIndex &nifi) const
        {
            return nifi.node_id == node_id && nifi.feature_index == feature_index;
        }
    };

    struct NodeIdFeatureIndexHash
    {
        size_t operator()(const NodeIdFeatureIndex &nifi) const
        {
            return nifi.node_id ^ nifi.feature_index;
        }
    };
    std::unordered_map<NodeIdFeatureIndex, std::vector<std::pair<TrackLink, NodeIdFeatureIndex>>,
                       NodeIdFeatureIndexHash>
        _edge_id_feature_index_tracklinks;
};

} // namespace opencalibration
