#pragma once

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geometry/intersection.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>

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
        Eigen::Vector3d *loc_ptr;
        Eigen::Quaterniond *rot_ptr;
        bool optimize = true;
        size_t node_id;
    } source, dest;
};

class RelaxProblem
{
  public:
    RelaxProblem();

    void initialize(std::vector<NodePose> &nodes);
    bool shouldAddEdgeToOptimization(const std::unordered_set<size_t> &edges_to_optimize, size_t edge_id,
                                     const MeasurementGraph::Edge &edge);

    OptimizationPackage::PoseOpt nodeid2poseopt(const MeasurementGraph &graph, size_t node_id);

    void addRelationCost(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge);

    void addMeasurementsCost(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge);

    void addDownwardsPrior();

    void solve();

    ceres::Problem::Options problemOptions;
    ceres::Solver::Options solverOptions;
    std::unique_ptr<ceres::HuberLoss> huber_loss;

  private:
    ceres::EigenQuaternionParameterization _quat_parameterization;
    std::unique_ptr<ceres::Problem> _problem;

    ceres::Solver::Summary _summary;
    ceres::Solver _solver;

    std::unordered_map<size_t, NodePose *> _nodes_to_optimize;
    std::unordered_set<size_t> _edges_used;

    using VecVec3D = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
    std::unordered_map<size_t, VecVec3D> _edge_points;
};

} // namespace opencalibration
