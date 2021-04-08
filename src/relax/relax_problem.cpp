#include <opencalibration/relax/relax_problem.hpp>

#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{

RelaxProblem::RelaxProblem() : huber_loss(new ceres::HuberLoss(M_PI_2))
{
    problemOptions.cost_function_ownership = ceres::TAKE_OWNERSHIP;
    problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problemOptions.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem.reset(new ceres::Problem(problemOptions));

    solverOptions.num_threads = 1;
    solverOptions.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solverOptions.max_num_iterations = 150;
}

void RelaxProblem::initialize(std::vector<NodePose> &nodes)
{
    nodes_to_optimize.reserve(nodes.size());

    for (NodePose &n : nodes)
    {
        n.orientation.normalize();
        nodes_to_optimize.emplace(n.node_id, &n);
    }
}

bool RelaxProblem::shouldOptimizeEdge(const std::unordered_set<size_t> &edges_to_optimize, size_t edge_id,
                                      const MeasurementGraph::Edge &edge)
{
    // skip edges not whitelisted
    if (edges_to_optimize.find(edge_id) == edges_to_optimize.end())
    {
        return false;
    }

    // skip edges already used in this optimization problem
    if (edges_used.find(edge_id) != edges_used.end())
    {
        return false;
    }

    // skip unitialized edges
    if (edge.payload.relative_rotation.coeffs().hasNaN() || edge.payload.relative_translation.hasNaN())
    {
        return false;
    }

    return true;
}

OptimizationPackage::PoseOpt RelaxProblem::nodeid2poseopt(const MeasurementGraph &graph, size_t node_id)
{
    OptimizationPackage::PoseOpt po;
    po.node_id = node_id;
    auto opt_iter = nodes_to_optimize.find(node_id);
    if (opt_iter != nodes_to_optimize.end())
    {
        NodePose *other = opt_iter->second;
        po.loc_ptr = &other->position;
        po.rot_ptr = &other->orientation;
        po.optimize = true;
    }
    else
    {
        const MeasurementGraph::Node *other = graph.getNode(node_id);
        if (other == nullptr)
        {
            spdlog::error("Null node referenced from edge list");
            po.loc_ptr = nullptr;
            po.rot_ptr = nullptr;
            po.optimize = false;
        }
        else if (other->payload.orientation.coeffs().hasNaN() || other->payload.position.hasNaN())
        {
            po.loc_ptr = nullptr;
            po.rot_ptr = nullptr;
            po.optimize = false;
        }
        else
        {
            po.loc_ptr = const_cast<Eigen::Vector3d *>(&other->payload.position);
            po.rot_ptr = const_cast<Eigen::Quaterniond *>(&other->payload.orientation);
            po.optimize = false;
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

    using CostFunction = ceres::AutoDiffCostFunction<DecomposedRotationCost, 3, 4, 4>;
    std::unique_ptr<CostFunction> func(
        new CostFunction(new DecomposedRotationCost(*pkg.relations, pkg.source.loc_ptr, pkg.dest.loc_ptr)));

    // test-evaluate the cost function to make sure no weird data gets into the relaxation
    Eigen::Matrix<double, 4, 3> jac[2];
    jac[0].setConstant(NAN);
    jac[1].setConstant(NAN);
    double *jacdata[2] = {jac[0].data(), jac[1].data()};
    Eigen::Vector3d res;
    res.setConstant(NAN);
    double *datas[2] = {pkg.source.rot_ptr->coeffs().data(), pkg.dest.rot_ptr->coeffs().data()};
    bool success = func->Evaluate(datas, res.data(), jacdata);

    if (!success || !res.allFinite() || !jac[0].allFinite() || !jac[1].allFinite())
    {
        spdlog::warn("Bad camera relation prevented from entering minimization");
        return;
    }

    problem->AddResidualBlock(func.release(), huber_loss.get(), datas[0], datas[1]);
    problem->SetParameterization(datas[0], &quat_parameterization);
    problem->SetParameterization(datas[1], &quat_parameterization);

    if (!pkg.source.optimize)
    {
        constant_nodes.emplace(pkg.source.node_id);
    }
    if (!pkg.dest.optimize)
    {
        constant_nodes.emplace(pkg.dest.node_id);
    }

    edges_used.emplace(edge_id);
}

void RelaxProblem::addMeasurementsCost(const MeasurementGraph &graph, size_t edge_id,
                                       const MeasurementGraph::Edge &edge)
{
    auto &points = edge_points[edge_id];
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

    for (const auto &inlier : edge.payload.inlier_matches)
    {
        // make 3D intersection, add it to `points`
        Eigen::Vector3d source_ray = source_rot * image_to_3d(inlier.pixel_1, source_model);
        Eigen::Vector3d dest_ray = dest_rot * image_to_3d(inlier.pixel_2, dest_model);
        Eigen::Vector4d intersection = rayIntersection(*pkg.source.loc_ptr, source_ray, *pkg.dest.loc_ptr, dest_ray);

        points.emplace_back(intersection.topRows<3>());

        // add cost functions for this 3D point from both the source and dest camera
        using CostFunction = ceres::AutoDiffCostFunction<PixelErrorCost, 2, 4, 3>;

        problem->AddResidualBlock(
            new CostFunction(new PixelErrorCost(*pkg.source.loc_ptr, source_model, inlier.pixel_1)), huber_loss.get(),
            pkg.source.rot_ptr->coeffs().data(), points.back().data());
        problem->SetParameterization(pkg.source.rot_ptr->coeffs().data(), &quat_parameterization);

        problem->AddResidualBlock(new CostFunction(new PixelErrorCost(*pkg.dest.loc_ptr, dest_model, inlier.pixel_2)),
                                  huber_loss.get(), pkg.dest.rot_ptr->coeffs().data(), points.back().data());
        problem->SetParameterization(pkg.dest.rot_ptr->coeffs().data(), &quat_parameterization);
    }

    if (!pkg.source.optimize)
    {
        constant_nodes.emplace(pkg.source.node_id);
    }
    if (!pkg.dest.optimize)
    {
        constant_nodes.emplace(pkg.dest.node_id);
    }

    edges_used.emplace(edge_id);
}

void RelaxProblem::addDownwardsPrior(std::vector<NodePose> &nodes)
{
    for (NodePose &n : nodes)
    {
        problem->AddResidualBlock(
            new ceres::AutoDiffCostFunction<PointsDownwardsPrior, 1, 4>(new PointsDownwardsPrior()), nullptr,
            n.orientation.coeffs().data());
        problem->SetParameterization(n.orientation.coeffs().data(), &quat_parameterization);
    }
}

void RelaxProblem::setConstantBlocks(const MeasurementGraph &graph)
{

    for (size_t const_node_id : constant_nodes)
    {
        problem->SetParameterBlockConstant(
            const_cast<double *>(graph.getNode(const_node_id)->payload.orientation.coeffs().data()));
    }
}

void RelaxProblem::solve(std::vector<NodePose> &nodes)
{
    spdlog::debug("Start rotation relax: {} active nodes, {} edges, {} inactive nodes", nodes.size(), edges_used.size(),
                  constant_nodes.size());

    solver.Solve(solverOptions, problem.get(), &summary);
    spdlog::info(summary.BriefReport());
    spdlog::debug(summary.FullReport());

    for (NodePose &n : nodes)
    {
        n.orientation.normalize();
    }
}

} // namespace opencalibration