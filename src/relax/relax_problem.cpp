#include <opencalibration/relax/relax_problem.hpp>

#include <opencalibration/relax/relax_cost_function.hpp>

#include <glog/log_severity.h>

namespace
{

class SpdLogSink : public google::LogSink
{
  public:
    void send(google::LogSeverity severity, const char *full_filename, const char *base_filename, int line,
              const struct ::tm *tm_time, const char *message, size_t message_len) override
    {
        // unused, maybe integrate them somehow?
        (void)full_filename;
        (void)base_filename;
        (void)line;
        (void)tm_time;

        spdlog::level::level_enum level;
        switch (severity)
        {
        case google::GLOG_INFO:
            level = spdlog::level::debug;
            break;
        case google::GLOG_WARNING:
            level = spdlog::level::info;
            break;
        case google::GLOG_ERROR:
            level = spdlog::level::warn;
            break;
        case google::GLOG_FATAL:
            level = spdlog::level::err;
            break;
        default:
            level = spdlog::level::critical;
        }
        spdlog::log(level, "{}", std::string_view(message, message_len));
    }
};

struct RegisterCeresLogger
{
    SpdLogSink sink;

    RegisterCeresLogger()
    {
        google::InitGoogleLogging("opencalibration");
        google::AddLogSink(&sink);
    }
};

static RegisterCeresLogger *__init_at_load = new RegisterCeresLogger();
} // namespace

namespace opencalibration
{

RelaxProblem::RelaxProblem() : huber_loss(new ceres::HuberLoss(M_PI_2))
{
    _problemOptions.cost_function_ownership = ceres::TAKE_OWNERSHIP;
    _problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    _problemOptions.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    _problem.reset(new ceres::Problem(_problemOptions));

    solverOptions.num_threads = 1;
    solverOptions.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solverOptions.max_num_iterations = 250;
    solverOptions.use_nonmonotonic_steps = true;
    solverOptions.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    solverOptions.dense_linear_algebra_library_type = ceres::EIGEN;
    solverOptions.initial_trust_region_radius = 1;
    solverOptions.logging_type = ceres::SILENT;
}

void RelaxProblem::initialize(std::vector<NodePose> &nodes)
{
    _nodes_to_optimize.reserve(nodes.size());

    for (NodePose &n : nodes)
    {
        n.orientation.normalize();
        _nodes_to_optimize.emplace(n.node_id, &n);
    }
}

bool RelaxProblem::shouldAddEdgeToOptimization(const std::unordered_set<size_t> &edges_to_optimize, size_t edge_id)
{
    // skip edges not whitelisted
    if (edges_to_optimize.find(edge_id) == edges_to_optimize.end())
    {
        return false;
    }

    // skip edges already used in this optimization problem
    if (_edges_used.find(edge_id) != _edges_used.end())
    {
        return false;
    }

    return true;
}

OptimizationPackage::PoseOpt RelaxProblem::nodeid2poseopt(const MeasurementGraph &graph, size_t node_id)
{
    OptimizationPackage::PoseOpt po;
    po.node_id = node_id;
    auto opt_iter = _nodes_to_optimize.find(node_id);
    if (opt_iter != _nodes_to_optimize.end())
    {
        po.optimize = true;

        NodePose *np = opt_iter->second;
        po.loc_ptr = &np->position;
        po.rot_ptr = &np->orientation;
    }
    else
    {
        po.optimize = false;

        const MeasurementGraph::Node *node = graph.getNode(node_id);
        if (node != nullptr && node->payload.orientation.coeffs().allFinite() && node->payload.position.allFinite())
        {
            // const_cast these, but mark as "don't optimize" so that they don't get changed downstream
            po.loc_ptr = const_cast<Eigen::Vector3d *>(&node->payload.position);
            po.rot_ptr = const_cast<Eigen::Quaterniond *>(&node->payload.orientation);
        }
    }
    return po;
}

void RelaxProblem::addRelationCost(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge)
{
    OptimizationPackage pkg;
    pkg.relations = &edge.payload;
    pkg.source = nodeid2poseopt(graph, edge.getSource());
    pkg.dest = nodeid2poseopt(graph, edge.getDest());

    if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
        return;

    if (!pkg.source.rot_ptr->coeffs().allFinite() || !pkg.dest.rot_ptr->coeffs().allFinite())
        return;

    if (!pkg.source.loc_ptr->allFinite() || !pkg.dest.loc_ptr->allFinite())
        return;

    using CostFunction =
        ceres::AutoDiffCostFunction<DualDecomposedRotationCost, DualDecomposedRotationCost::NUM_RESIDUALS,
                                    DualDecomposedRotationCost::NUM_PARAMETERS_1,
                                    DualDecomposedRotationCost::NUM_PARAMETERS_2>;
    std::unique_ptr<CostFunction> func(
        new CostFunction(new DualDecomposedRotationCost(*pkg.relations, pkg.source.loc_ptr, pkg.dest.loc_ptr)));

    double *datas[2] = {pkg.source.rot_ptr->coeffs().data(), pkg.dest.rot_ptr->coeffs().data()};

    _problem->AddResidualBlock(func.release(), huber_loss.get(), datas[0], datas[1]);
    _problem->SetParameterization(datas[0], &_quat_parameterization);
    _problem->SetParameterization(datas[1], &_quat_parameterization);

    if (!pkg.source.optimize)
    {
        _problem->SetParameterBlockConstant(datas[0]);
    }
    if (!pkg.dest.optimize)
    {
        _problem->SetParameterBlockConstant(datas[1]);
    }

    _edges_used.emplace(edge_id);
}

void RelaxProblem::addGlobalPlaneMeasurementsCost(const MeasurementGraph &graph, size_t edge_id,
                                                  const MeasurementGraph::Edge &edge)
{
    auto &points = _edge_points[edge_id];
    points.reserve(edge.payload.inlier_matches.size());

    OptimizationPackage pkg;
    pkg.relations = &edge.payload;
    pkg.source = nodeid2poseopt(graph, edge.getSource());
    pkg.dest = nodeid2poseopt(graph, edge.getDest());

    if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
        return;

    const auto &source_model = graph.getNode(edge.getSource())->payload.model;
    const auto &dest_model = graph.getNode(edge.getDest())->payload.model;

    //     Eigen::Matrix3d source_rot = pkg.source.rot_ptr->toRotationMatrix();
    //     Eigen::Matrix3d dest_rot = pkg.dest.rot_ptr->toRotationMatrix();

    double *datas[5] = {pkg.source.rot_ptr->coeffs().data(), pkg.dest.rot_ptr->coeffs().data(),
                        &_global_plane.corner[0].z(), &_global_plane.corner[1].z(), &_global_plane.corner[2].z()};
    Eigen::Vector2d corner2d[3]{_global_plane.corner[0].topRows<2>(), _global_plane.corner[1].topRows<2>(),
                                _global_plane.corner[2].topRows<2>()};
    for (const auto &inlier : edge.payload.inlier_matches)
    {
        // make 3D intersection, add it to `points`
        Eigen::Vector3d source_ray = image_to_3d(inlier.pixel_1, source_model);
        Eigen::Vector3d dest_ray = image_to_3d(inlier.pixel_2, dest_model);

        // add cost functions for this 3D point from both the source and dest camera
        using CostFunction = ceres::AutoDiffCostFunction<
            PlaneIntersectionAngleCost, PlaneIntersectionAngleCost::NUM_RESIDUALS,
            PlaneIntersectionAngleCost::NUM_PARAMETERS_1, PlaneIntersectionAngleCost::NUM_PARAMETERS_2,
            PlaneIntersectionAngleCost::NUM_PARAMETERS_3, PlaneIntersectionAngleCost::NUM_PARAMETERS_4,
            PlaneIntersectionAngleCost::NUM_PARAMETERS_5>;

        std::unique_ptr<CostFunction> func(new CostFunction(new PlaneIntersectionAngleCost(
            *pkg.source.loc_ptr, *pkg.dest.loc_ptr, source_ray, dest_ray, corner2d[0], corner2d[1], corner2d[2])));

        _problem->AddResidualBlock(func.release(), huber_loss.get(), datas[0], datas[1], datas[2], datas[3], datas[4]);

        _problem->SetParameterization(datas[0], &_quat_parameterization);
        _problem->SetParameterization(datas[1], &_quat_parameterization);
    }

    if (!pkg.source.optimize)
    {
        _problem->SetParameterBlockConstant(datas[0]);
    }
    if (!pkg.dest.optimize)
    {
        _problem->SetParameterBlockConstant(datas[1]);
    }

    _edges_used.emplace(edge_id);
}

void RelaxProblem::addPointMeasurementsCost(const MeasurementGraph &graph, size_t edge_id,
                                            const MeasurementGraph::Edge &edge)
{
    auto &points = _edge_points[edge_id];
    points.reserve(edge.payload.inlier_matches.size());

    OptimizationPackage pkg;
    pkg.relations = &edge.payload;
    pkg.source = nodeid2poseopt(graph, edge.getSource());
    pkg.dest = nodeid2poseopt(graph, edge.getDest());

    if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
        return;

    const auto &source_model = graph.getNode(edge.getSource())->payload.model;
    const auto &dest_model = graph.getNode(edge.getDest())->payload.model;

    Eigen::Matrix3d source_rot = pkg.source.rot_ptr->toRotationMatrix();
    Eigen::Matrix3d dest_rot = pkg.dest.rot_ptr->toRotationMatrix();

    double *datas[2] = {pkg.source.rot_ptr->coeffs().data(), pkg.dest.rot_ptr->coeffs().data()};
    for (const auto &inlier : edge.payload.inlier_matches)
    {
        // make 3D intersection, add it to `points`
        Eigen::Vector3d source_ray = source_rot * image_to_3d(inlier.pixel_1, source_model);
        Eigen::Vector3d dest_ray = dest_rot * image_to_3d(inlier.pixel_2, dest_model);
        Eigen::Vector4d intersection = rayIntersection(*pkg.source.loc_ptr, source_ray, *pkg.dest.loc_ptr, dest_ray);

        points.emplace_back(intersection.topRows<3>());

        // add cost functions for this 3D point from both the source and dest camera
        using CostFunction =
            ceres::AutoDiffCostFunction<PixelErrorCost, PixelErrorCost::NUM_RESIDUALS, PixelErrorCost::NUM_PARAMETERS_1,
                                        PixelErrorCost::NUM_PARAMETERS_2>;

        std::unique_ptr<CostFunction> func[2]{
            std::make_unique<CostFunction>(new PixelErrorCost(*pkg.source.loc_ptr, source_model, inlier.pixel_1)),
            std::make_unique<CostFunction>(new PixelErrorCost(*pkg.dest.loc_ptr, dest_model, inlier.pixel_2))};

        for (int i = 0; i < 2; i++)
        {
            _problem->AddResidualBlock(func[i].release(), huber_loss.get(), datas[i], points.back().data());
            _problem->SetParameterization(datas[i], &_quat_parameterization);
        }
    }

    if (!pkg.source.optimize)
    {
        _problem->SetParameterBlockConstant(datas[0]);
    }
    if (!pkg.dest.optimize)
    {
        _problem->SetParameterBlockConstant(datas[1]);
    }

    _edges_used.emplace(edge_id);
}

void RelaxProblem::initializeGroundPlane()
{
    // calculate average height and xy bounding box
    Eigen::Vector2d xy_min(1e12, 1e12), xy_max(-1e12, -1e12);
    double height = 0;

    for (const auto &p : _nodes_to_optimize)
    {
        Eigen::Vector3d &loc = p.second->position;
        xy_min = xy_min.cwiseMin(loc.topRows<2>());
        xy_max = xy_max.cwiseMax(loc.topRows<2>());
        height += loc.z();
    }
    height /= _nodes_to_optimize.size();

    // place triangle to enclose bounding box entirely, 100m below avg height
    /*             A
     *            / \
     *           /   \
     *          /     \
     *         /._____.\
     *        / |     | \
     *       /  |     |  \
     *      /   |_____|   \
     *     /_______________\
     *    B                 C
     *
     *    BC = 2 * (max(width, height) + margin)
     *    A-BC = BC
     *
     */

    constexpr double margin = 50;
    height -= margin;
    Eigen::Vector2d center = (xy_min + xy_max) / 2;
    const double spacing = (xy_max - xy_min).maxCoeff() + margin;

    _global_plane.corner[0] << Eigen::Vector2d(-spacing, -spacing) + center, height;
    _global_plane.corner[1] << Eigen::Vector2d(spacing, -spacing) + center, height;
    _global_plane.corner[2] << Eigen::Vector2d(0, spacing) + center, height;
}

void RelaxProblem::addDownwardsPrior()
{
    for (auto &p : _nodes_to_optimize)
    {
        double *d = p.second->orientation.coeffs().data();
        using CostFunction = ceres::AutoDiffCostFunction<PointsDownwardsPrior, PointsDownwardsPrior::NUM_RESIDUALS,
                                                         PointsDownwardsPrior::NUM_PARAMETERS_1>;
        _problem->AddResidualBlock(new CostFunction(new PointsDownwardsPrior()), nullptr, d);
        _problem->SetParameterization(d, &_quat_parameterization);
    }
}

void RelaxProblem::solve()
{
    spdlog::debug("Start rotation relax: {} active nodes, {} edges", _nodes_to_optimize.size(), _edges_used.size());

    _solver.Solve(solverOptions, _problem.get(), &_summary);
    spdlog::info(_summary.BriefReport());
    spdlog::debug(_summary.FullReport());

    for (auto &p : _nodes_to_optimize)
    {
        p.second->orientation.normalize();
    }
}

} // namespace opencalibration
