#include <opencalibration/relax/relax_problem.hpp>

#include <opencalibration/relax/autodiff_cost_function.hpp>
#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/surface/intersect.hpp>

#include "ceres_log_forwarding.cpp.inc"

#include <thread>

namespace opencalibration
{

RelaxProblem::RelaxProblem()
    : _loss(new ceres::TrivialLoss(), ceres::TAKE_OWNERSHIP), _brown2_parameterization(3, {1, 2}),
      _brown24_parameterization(3, {2}), _brown246_parameterization(3)
{
    _problemOptions.cost_function_ownership = ceres::TAKE_OWNERSHIP;
    _problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    _problemOptions.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    _problem.reset(new ceres::Problem(_problemOptions));

    _solver_options.num_threads = 1;
    _solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    _solver_options.max_num_iterations = 500;
    _solver_options.use_nonmonotonic_steps = true;
    _solver_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    _solver_options.dense_linear_algebra_library_type = ceres::EIGEN;
    _solver_options.initial_trust_region_radius = 1;
    _solver_options.logging_type = ceres::SILENT;
}

void RelaxProblem::setupDecompositionProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                             const std::unordered_set<size_t> &edges_to_optimize)
{

    _loss.Reset(new ceres::HuberLoss(10 * M_PI / 180), ceres::TAKE_OWNERSHIP);
    std::unordered_map<size_t, CameraModel> cam_models;
    initialize(nodes, cam_models);
    _solver_options.initial_trust_region_radius = 0.1;

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            addRelationCost(graph, edge_id, *edge);
        }
    }

    addDownwardsPrior();
}

void RelaxProblem::setupGroundPlaneProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                           std::unordered_map<size_t, CameraModel> &cam_models,
                                           const std::unordered_set<size_t> &edges_to_optimize,
                                           const RelaxOptionSet &options)
{
    initialize(nodes, cam_models);
    initializeGroundPlane();
    _loss.Reset(new ceres::HuberLoss(1 * M_PI / 180), ceres::TAKE_OWNERSHIP);
    gridFilterMatchesPerImage(graph, edges_to_optimize, 0.15);

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            addRayTriangleMeasurementCost(graph, edge_id, *edge, options);
        }
    }

    addDownwardsPrior();
}

void RelaxProblem::setupGroundMeshProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                          std::unordered_map<size_t, CameraModel> &cam_models,
                                          const std::unordered_set<size_t> &edges_to_optimize,
                                          const RelaxOptionSet &options,
                                          const std::vector<surface_model> &previousSurfaces)
{
    initialize(nodes, cam_models);
    initializeGroundMesh(previousSurfaces);
    _loss.Reset(new ceres::HuberLoss(1 * M_PI / 180), ceres::TAKE_OWNERSHIP);
    gridFilterMatchesPerImage(graph, edges_to_optimize, 0.05);

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            addRayTriangleMeasurementCost(graph, edge_id, *edge, options);
        }
    }

    addMeshFlatPrior();
}

void RelaxProblem::setup3dPointProblem(const MeasurementGraph &graph, std::vector<opencalibration::NodePose> &nodes,
                                       std::unordered_map<size_t, opencalibration::CameraModel> &cam_models,
                                       const std::unordered_set<size_t> &edges_to_optimize,
                                       const RelaxOptionSet &options)
{
    initialize(nodes, cam_models);
    _loss.Reset(new ceres::HuberLoss(10), ceres::TAKE_OWNERSHIP);

    gridFilterMatchesPerImage(graph, edges_to_optimize, 0.05);

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            addPointMeasurementsCost(graph, edge_id, *edge, options);
        }
    }

    _solver_options.max_num_iterations = 300;
    _solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
}

void RelaxProblem::initialize(std::vector<NodePose> &nodes, std::unordered_map<size_t, CameraModel> &cam_models)
{
    _nodes_to_optimize.reserve(nodes.size());
    for (NodePose &n : nodes)
    {
        _nodes_to_optimize.emplace(n.node_id, &n);
    }

    _cam_models_to_optimize.reserve(cam_models.size());
    for (auto &id_model : cam_models)
    {
        _cam_models_to_optimize[id_model.first] = &id_model.second;
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

OptimizationPackage::PoseOpt RelaxProblem::nodeid2poseopt(const MeasurementGraph &graph, size_t node_id,
                                                          bool load_cam_model)
{
    OptimizationPackage::PoseOpt po;
    po.node_id = node_id;
    auto opt_iter = _nodes_to_optimize.find(node_id);
    const MeasurementGraph::Node *node = graph.getNode(node_id);
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

        if (node != nullptr && node->payload.orientation.coeffs().allFinite() && node->payload.position.allFinite())
        {
            // const_cast these, but mark as "don't optimize" so that they don't get changed downstream
            po.loc_ptr = const_cast<Eigen::Vector3d *>(&node->payload.position);
            po.rot_ptr = const_cast<Eigen::Quaterniond *>(&node->payload.orientation);
        }
    }
    if (load_cam_model)
    {
        if (node != nullptr)
        {
            auto model_iter = _cam_models_to_optimize.find(node->payload.model->id);
            if (model_iter == _cam_models_to_optimize.end())
            {
                if (po.optimize)
                {
                    spdlog::warn("Trying to optimize camera without mutable model");
                }
                po.model_ptr = node->payload.model.get();
            }
            else
            {
                po.model_ptr = model_iter->second;
            }
        }
        else
        {
            spdlog::warn("Need to get camera model from unknown node id {}", node_id);
        }
    }

    return po;
}

void RelaxProblem::gridFilterMatchesPerImage(const MeasurementGraph &graph,
                                             const std::unordered_set<size_t> &edges_to_optimize,
                                             double grid_cell_image_fraction)
{
    for (size_t edge_id : edges_to_optimize)
    {
        auto *edge_ptr = graph.getEdge(edge_id);
        if (edge_ptr == nullptr)
            continue;

        const MeasurementGraph::Edge &edge = *edge_ptr;

        OptimizationPackage pkg;
        pkg.relations = &edge.payload;
        pkg.source = nodeid2poseopt(graph, edge.getSource());
        pkg.dest = nodeid2poseopt(graph, edge.getDest());

        if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
            return;

        const auto &source_model = *graph.getNode(edge.getSource())->payload.model;
        const auto &dest_model = *graph.getNode(edge.getDest())->payload.model;

        Eigen::Matrix3d source_rot = pkg.source.rot_ptr->toRotationMatrix();
        Eigen::Matrix3d dest_rot = pkg.dest.rot_ptr->toRotationMatrix();

        auto &source_filter = _grid_filter[edge.getSource()][edge_id];
        auto &dest_filter = _grid_filter[edge.getDest()][edge_id];
        source_filter.setResolution(grid_cell_image_fraction);
        dest_filter.setResolution(grid_cell_image_fraction);

        for (const auto &inlier : edge.payload.inlier_matches)
        {
            // make 3D intersection, add it to `points`
            ray_d source_ray = {source_rot * image_to_3d(inlier.pixel_1, source_model), *pkg.source.loc_ptr};
            ray_d dest_ray = {dest_rot * image_to_3d(inlier.pixel_2, dest_model), *pkg.dest.loc_ptr};
            auto intersection = rayIntersection(source_ray, dest_ray);

            const double score = intersection.second < 0 ? 0. : 1. / (1. + intersection.second);
            if (score > 0)
            {
                source_filter.addMeasurement(inlier.pixel_1.x() / source_model.pixels_cols,
                                             inlier.pixel_1.y() / source_model.pixels_rows, score, &inlier);
                dest_filter.addMeasurement(inlier.pixel_2.x() / dest_model.pixels_cols,
                                           inlier.pixel_2.y() / dest_model.pixels_rows, score, &inlier);
            }
        }
    }
}

void RelaxProblem::addRelationCost(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge)
{
    if (edge.payload.inlier_matches.size() == 0)
        return;

    OptimizationPackage pkg;
    pkg.relations = &edge.payload;
    pkg.source = nodeid2poseopt(graph, edge.getSource(), false);
    pkg.dest = nodeid2poseopt(graph, edge.getDest(), false);

    if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
        return;

    if (!pkg.source.rot_ptr->coeffs().allFinite() || !pkg.dest.rot_ptr->coeffs().allFinite())
        return;

    if (!pkg.source.loc_ptr->allFinite() || !pkg.dest.loc_ptr->allFinite())
        return;

    std::unique_ptr<ceres::CostFunction> func(
        newAutoDiffMultiDecomposedRotationCost(*pkg.relations, pkg.source.loc_ptr, pkg.dest.loc_ptr));

    double *datas[2] = {pkg.source.rot_ptr->coeffs().data(), pkg.dest.rot_ptr->coeffs().data()};

    _problem->AddResidualBlock(func.release(), &_loss, datas[0], datas[1]);
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

void RelaxProblem::addRayTriangleMeasurementCost(const MeasurementGraph &graph, size_t edge_id,
                                                 const MeasurementGraph::Edge &edge, const RelaxOptionSet &options)
{
    auto &points = _edge_tracks[edge_id];
    points.reserve(edge.payload.inlier_matches.size());

    OptimizationPackage pkg;
    pkg.relations = &edge.payload;
    pkg.source = nodeid2poseopt(graph, edge.getSource());
    pkg.dest = nodeid2poseopt(graph, edge.getDest());

    if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
        return;

    const auto &source_model = *pkg.source.model_ptr;
    const auto &dest_model = *pkg.dest.model_ptr;

    const auto &source_whitelist = _grid_filter[edge.getSource()][edge_id].getBestMeasurementsPerCell();
    const auto &dest_whitelist = _grid_filter[edge.getDest()][edge_id].getBestMeasurementsPerCell();

    const std::array<double *, 2> datas = {pkg.source.rot_ptr->coeffs().data(), pkg.dest.rot_ptr->coeffs().data()};

    MeshIntersectionSearcher intersectionSearcher;
    intersectionSearcher.init(_mesh);

    bool points_added = false;
    for (const auto &inlier : edge.payload.inlier_matches)
    {
        if (source_whitelist.find(&inlier) == source_whitelist.end() &&
            dest_whitelist.find(&inlier) == dest_whitelist.end())
            continue;

        // make 3D intersection, add it to `points`
        ray_d sourceRay, destRay;
        sourceRay.dir = image_to_3d(inlier.pixel_1, source_model);
        sourceRay.offset = *pkg.source.loc_ptr;
        destRay.dir = image_to_3d(inlier.pixel_2, dest_model);
        destRay.offset = *pkg.dest.loc_ptr;

        auto sourceDestIntersection = rayIntersection(ray_d{*pkg.source.rot_ptr * sourceRay.dir, sourceRay.offset},
                                                      ray_d{*pkg.dest.rot_ptr * destRay.dir, destRay.offset});

        const auto intersectionTriangle =
            intersectionSearcher.triangleIntersect(ray_d{Eigen::Vector3d(0, 0, 1), sourceDestIntersection.first});
        if (intersectionTriangle.type != MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
        {
            continue;
        }

        const auto &triangle = intersectionTriangle.nodeLocations;

        std::array<Eigen::Vector2d, 3> corner2d = {triangle[0]->topRows<2>(), triangle[1]->topRows<2>(),
                                                   triangle[2]->topRows<2>()};

        std::array<const double *, 3> constZValues{&triangle[0]->z(), &triangle[1]->z(), &triangle[2]->z()};
        std::array<double *, 3> zValues;
        for (size_t i = 0; i < 3; i++)
        {
            zValues[i] = const_cast<double *>(constZValues[i]);
        }

        // TODO: select correct cost function based on options
        (void)options;

        // add cost functions for this 3D point from both the source and dest camera
        std::unique_ptr<ceres::CostFunction> func(newAutoDiffPlaneIntersectionAngleCost(
            sourceRay.offset, destRay.offset, sourceRay.dir, destRay.dir, corner2d[0], corner2d[1], corner2d[2]));

        _problem->AddResidualBlock(func.release(), &_loss, datas[0], datas[1], zValues[0], zValues[1], zValues[2]);
        points_added = true;
    }
    if (points_added)
    {
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
    }

    _edges_used.insert(edge_id);
}

void RelaxProblem::relaxObservedModelOnly()
{
    // optimize just the 3d points to start, since we don't do proper triangulation

    std::vector<double *> params;
    _problem->GetParameterBlocks(&params);

    std::vector<bool> isConst;
    isConst.reserve(params.size());
    std::unordered_set<double *> params_set;
    for (double *p : params)
    {
        isConst.push_back(_problem->IsParameterBlockConstant(p));
        _problem->SetParameterBlockConstant(p);
        params_set.insert(p);
    }
    for (auto &et : _edge_tracks)
    {
        for (auto &t : et.second)
        {
            if (params_set.find(t.point.data()) != params_set.end())
            {
                _problem->SetParameterBlockVariable(t.point.data());
            }
        }
    }

    for (auto iter = _mesh.nodebegin(); iter != _mesh.nodeend(); ++iter)
    {
        auto &p = iter->second.payload.location;
        if (params_set.find(&p.z()) != params_set.end())
            _problem->SetParameterBlockVariable(&p.z());
    }

    spdlog::debug("optimizing surface only");
    solve();
    for (size_t i = 0; i < params.size(); i++)
    {
        if (isConst[i])
        {
            _problem->SetParameterBlockConstant(params[i]);
        }
        else
        {
            _problem->SetParameterBlockVariable(params[i]);
        }
    }
}

void RelaxProblem::addPointMeasurementsCost(const MeasurementGraph &graph, size_t edge_id,
                                            const MeasurementGraph::Edge &edge, const RelaxOptionSet &options)
{
    auto &points = _edge_tracks[edge_id];
    points.reserve(edge.payload.inlier_matches.size());

    OptimizationPackage pkg;
    pkg.relations = &edge.payload;
    pkg.source = nodeid2poseopt(graph, edge.getSource());
    pkg.dest = nodeid2poseopt(graph, edge.getDest());

    if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
        return;

    if (options.hasAll({Option::FOCAL_LENGTH}) && (pkg.source.model_ptr == nullptr || pkg.dest.model_ptr == nullptr))
        return;

    auto &source_model = *pkg.source.model_ptr;
    auto &dest_model = *pkg.dest.model_ptr;

    const auto &source_whitelist = _grid_filter[edge.getSource()][edge_id].getBestMeasurementsPerCell();
    const auto &dest_whitelist = _grid_filter[edge.getDest()][edge_id].getBestMeasurementsPerCell();

    double *orientation_ptrs[2] = {pkg.source.rot_ptr->coeffs().data(), pkg.dest.rot_ptr->coeffs().data()};
    bool points_added = false;
    for (const auto &inlier : edge.payload.inlier_matches)
    {

        if (source_whitelist.find(&inlier) == source_whitelist.end() &&
            dest_whitelist.find(&inlier) == dest_whitelist.end())
        {
            continue;
        }

        // make 3D intersection, add it to `points`
        auto intersection = rayIntersection(source_model, dest_model, *pkg.source.loc_ptr, *pkg.dest.loc_ptr,
                                            *pkg.source.rot_ptr, *pkg.dest.rot_ptr, inlier.pixel_1, inlier.pixel_2);
        NodeIdFeatureIndex nifi[2];
        nifi[0].node_id = edge.getSource();
        nifi[0].feature_index = inlier.feature_index_1;
        nifi[1].node_id = edge.getDest();
        nifi[1].feature_index = inlier.feature_index_2;
        points.emplace_back(FeatureTrack{intersection.first, intersection.second, {nifi[0], nifi[1]}});

        // add cost functions for this 3D point from both the source and dest camera
        std::unique_ptr<ceres::CostFunction> func[2];
        std::vector<double *> args[2];

        double *focals[2] = {&source_model.focal_length_pixels, &dest_model.focal_length_pixels};
        double *radials[2] = {source_model.radial_distortion.data(), dest_model.radial_distortion.data()};
        double *tangentials[2] = {source_model.tangential_distortion.data(), dest_model.tangential_distortion.data()};

        if (options.hasAll({Option::LENS_DISTORTIONS_TANGENTIAL, Option::LENS_DISTORTIONS_RADIAL, Option::FOCAL_LENGTH,
                            Option::ORIENTATION, Option::POINTS_3D}))
        {
            func[0].reset(newAutoDiffPixelErrorCost_OrientationFocalRadialTangential(*pkg.source.loc_ptr, source_model,
                                                                                     inlier.pixel_1));
            func[1].reset(newAutoDiffPixelErrorCost_OrientationFocalRadialTangential(*pkg.dest.loc_ptr, dest_model,
                                                                                     inlier.pixel_2));
            for (int i = 0; i < 2; i++)
            {
                args[i] = {orientation_ptrs[i], points.back().point.data(), focals[i], radials[i], tangentials[i]};
            }
        }
        else if (options.hasAll(
                     {Option::LENS_DISTORTIONS_RADIAL, Option::FOCAL_LENGTH, Option::ORIENTATION, Option::POINTS_3D}))
        {
            func[0].reset(
                newAutoDiffPixelErrorCost_OrientationFocalRadial(*pkg.source.loc_ptr, source_model, inlier.pixel_1));
            func[1].reset(
                newAutoDiffPixelErrorCost_OrientationFocalRadial(*pkg.dest.loc_ptr, dest_model, inlier.pixel_2));
            for (int i = 0; i < 2; i++)
            {
                args[i] = {orientation_ptrs[i], points.back().point.data(), focals[i], radials[i]};
            }
        }
        else if (options.hasAll({Option::FOCAL_LENGTH, Option::ORIENTATION, Option::POINTS_3D}))
        {
            func[0].reset(
                newAutoDiffPixelErrorCost_OrientationFocal(*pkg.source.loc_ptr, source_model, inlier.pixel_1));
            func[1].reset(newAutoDiffPixelErrorCost_OrientationFocal(*pkg.dest.loc_ptr, dest_model, inlier.pixel_2));

            for (int i = 0; i < 2; i++)
            {
                args[i] = {orientation_ptrs[i], points.back().point.data(), focals[i]};
            }
        }
        else if (options.hasAll({Option::ORIENTATION, Option::POINTS_3D}))
        {
            func[0].reset(newAutoDiffPixelErrorCost_Orientation(*pkg.source.loc_ptr, source_model, inlier.pixel_1));
            func[1].reset(newAutoDiffPixelErrorCost_Orientation(*pkg.dest.loc_ptr, dest_model, inlier.pixel_2));

            for (int i = 0; i < 2; i++)
            {
                args[i] = {orientation_ptrs[i], points.back().point.data()};
            }
        }
        else
        {
            spdlog::critical("No viable bundle options found");
        }

        bool all_finite = true;
        for (int i = 0; i < 2; i++)
        {
            Eigen::Vector2d res{NAN, NAN};
            func[i]->Evaluate(args[i].data(), res.data(), nullptr);

            if (!res.array().allFinite())
            {
                all_finite = false;
            }
        }
        if (!all_finite)
        {
            spdlog::trace("Skipping adding NaN track measurement residual");
            continue;
        }

        for (int i = 0; i < 2; i++)
        {
            _problem->AddResidualBlock(func[i].get(), &_loss, args[i]);
            func[i].release();
        }

        points_added = true;
    }

    if (points_added)
    {
        _problem->SetParameterization(orientation_ptrs[0], &_quat_parameterization);
        _problem->SetParameterization(orientation_ptrs[1], &_quat_parameterization);
        if (!pkg.source.optimize)
        {
            _problem->SetParameterBlockConstant(orientation_ptrs[0]);
        }
        if (!pkg.dest.optimize)
        {
            _problem->SetParameterBlockConstant(orientation_ptrs[1]);
        }
        if (options.hasAll({Option::LENS_DISTORTIONS_RADIAL}))
        {
            if (options.hasAll({Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION}))
            {
                _problem->SetParameterization(source_model.radial_distortion.data(), &_brown246_parameterization);
                _problem->SetParameterization(dest_model.radial_distortion.data(), &_brown246_parameterization);
            }
            else if (options.hasAll({Option::LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION}))
            {
                _problem->SetParameterization(source_model.radial_distortion.data(), &_brown24_parameterization);
                _problem->SetParameterization(dest_model.radial_distortion.data(), &_brown24_parameterization);
            }
            else if (options.hasAll({Option::LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION}))
            {
                _problem->SetParameterization(source_model.radial_distortion.data(), &_brown2_parameterization);
                _problem->SetParameterization(dest_model.radial_distortion.data(), &_brown2_parameterization);
            }
            else
            {
                spdlog::warn("No parameterization chosen for radial distortion");
            }
        }
    }

    _edges_used.emplace(edge_id);
}

void RelaxProblem::initializeGroundPlane()
{
    // calculate average height and xy bounding box
    Eigen::Vector2d xy_min(1e12, 1e12), xy_max(-1e12, -1e12);
    double height = 0;

    for (const auto &[key, value] : _nodes_to_optimize)
    {
        const Eigen::Vector3d &loc = value->position;
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

    // TODO: use a robust RANSAC based plane fit of the ray-ray intersections instead
    constexpr double margin = 50;
    height -= margin;
    Eigen::Vector2d center = (xy_min + xy_max) / 2;
    const double spacing = (xy_max - xy_min).maxCoeff() + margin;
    plane_3_corners_d plane;

    plane.corner[0] << Eigen::Vector2d(-spacing, -spacing) + center, height;
    plane.corner[1] << Eigen::Vector2d(spacing, -spacing) + center, height;
    plane.corner[2] << Eigen::Vector2d(0, spacing) + center, height;

    _mesh = MeshGraph();
    std::array<size_t, 3> nodeIds;
    for (size_t i = 0; i < 3; i++)
    {
        nodeIds[i] = _mesh.addNode(MeshNode{plane.corner[i]});
    }
    for (size_t i = 0; i < 3; i++)
    {
        _mesh.addEdge(MeshEdge{true, {nodeIds[(i + 2) % 3], 0}}, nodeIds[i], nodeIds[(i + 1) % 3]);
    }
}

void RelaxProblem::initializeGroundMesh(const std::vector<surface_model> &previousSurfaces)
{
    point_cloud cameraLocations;
    cameraLocations.reserve(_nodes_to_optimize.size());
    for (const auto &[key, value] : _nodes_to_optimize)
    {
        cameraLocations.push_back(value->position);
    }
    _mesh = rebuildMesh(cameraLocations, previousSurfaces);
}

void RelaxProblem::addDownwardsPrior()
{
    for (auto &p : _nodes_to_optimize)
    {
        if (!p.second->orientation.coeffs().hasNaN())
        {
            double *d = p.second->orientation.coeffs().data();
            _problem->AddResidualBlock(newAutoDiffPointsDownwardsPrior(1e-3), nullptr, d);
            _problem->SetParameterization(d, &_quat_parameterization);
        }
    }
}

void RelaxProblem::addMeshFlatPrior()
{

    // for each edge in the graph
    // add a cost for the nodes on each side being different heights
    for (auto iter = _mesh.edgebegin(); iter != _mesh.edgeend(); ++iter)
    {
        const size_t sourceId = iter->second.getSource();
        const size_t destId = iter->second.getDest();

        auto *sourceNode = _mesh.getNode(sourceId);
        auto *destNode = _mesh.getNode(destId);

        double *h1 = &sourceNode->payload.location.z();
        double *h2 = &destNode->payload.location.z();

        _problem->AddResidualBlock(newAutoDiffDifferenceCost(1e-4), nullptr, h1, h2);
    }
}

void RelaxProblem::solve()
{
    std::ostringstream thread_stream;
    thread_stream << std::this_thread::get_id();
    spdlog::info("Thread {} start relax: {} active nodes, {} edges, {} residuals", thread_stream.str(), _nodes_to_optimize.size(), _edges_used.size(), _problem->NumResiduals());

    _solver.Solve(_solver_options, _problem.get(), &_summary);
    spdlog::info("Thread {}  end relax iterations {}, cost ratio {}, time {}ms", thread_stream.str(), _summary.iterations.size(),
                 static_cast<float>(_summary.final_cost / _summary.initial_cost),
                 static_cast<float>(_summary.total_time_in_seconds*1000));
    spdlog::debug(_summary.FullReport());

    for (auto &p : _nodes_to_optimize)
    {
        p.second->orientation.normalize();
    }
}

surface_model RelaxProblem::getSurfaceModel()
{
    surface_model s;
    s.cloud.reserve(_edge_tracks.size());
    for (const auto &tv : _edge_tracks)
    {
        s.cloud.emplace_back();
        s.cloud.back().reserve(tv.second.size());
        for (const auto &t : tv.second)
        {
            s.cloud.back().push_back(t.point);
        }
    }

    s.mesh = _mesh;

    return s;
}

} // namespace opencalibration
