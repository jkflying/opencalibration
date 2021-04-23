#pragma once

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geometry/intersection.hpp>
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

class RelaxProblem
{
  public:
    RelaxProblem();

    void initialize(std::vector<NodePose> &nodes);
    bool shouldAddEdgeToOptimization(const std::unordered_set<size_t> &edges_to_optimize, size_t edge_id);

    OptimizationPackage::PoseOpt nodeid2poseopt(const MeasurementGraph &graph, size_t node_id);

    void addRelationCost(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge);

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

    ceres::Solver::Summary _summary;
    ceres::Solver _solver;

    std::unordered_map<size_t, NodePose *> _nodes_to_optimize;
    std::unordered_set<size_t> _edges_used;

    // Surface models
    plane_3_corners<double> _global_plane;
    std::unordered_map<size_t, point_cloud> _edge_points;
};

} // namespace opencalibration
