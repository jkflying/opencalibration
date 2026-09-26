#include <opencalibration/relax/relax_problem.hpp>

#include <Eigen/SparseCore>
#include <Eigen/SparseQR>
#include <opencalibration/relax/autodiff_cost_function.hpp>
#include <opencalibration/relax/relax_cost_function.hpp>

#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/surface/intersect.hpp>

#include "ceres_log_forwarding.cpp.inc"

#include <opencalibration/distort/invert_distortion.hpp>
#include <opencalibration/types/union_find.hpp>
#include <thread>

namespace opencalibration
{

constexpr double GPS_HORIZONTAL_SIGMA_METERS = 2.0;
constexpr double GPS_VERTICAL_SIGMA_METERS = 4.0;

RelaxProblem::RelaxProblem()
    : _loss(new ceres::TrivialLoss(), ceres::TAKE_OWNERSHIP),
      _pose_parameterization(ceres::EigenQuaternionManifold{}, ceres::EuclideanManifold<3>{}),
      _pose_fixed_position_parameterization(ceres::EigenQuaternionManifold{}, ceres::SubsetManifold(3, {0, 1, 2})),
      _brown2_parameterization(3, {1, 2}), _brown24_parameterization(3, {2}), _brown246_parameterization(3)
{
    _problemOptions.cost_function_ownership = ceres::TAKE_OWNERSHIP;
    _problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    _problemOptions.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    _problem.reset(new ceres::Problem(_problemOptions));

    _solver_options.num_threads = 1;
    _solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    _solver_options.max_num_iterations = 100;
    _solver_options.use_nonmonotonic_steps = false;
    _solver_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    _solver_options.dense_linear_algebra_library_type = ceres::EIGEN;
    _solver_options.initial_trust_region_radius = 1;
    _solver_options.logging_type = ceres::SILENT;
}

void RelaxProblem::setupDecompositionProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                             const ankerl::unordered_dense::set<size_t> &edges_to_optimize)
{

    _loss.Reset(new ceres::HuberLoss(10 * M_PI / 180), ceres::TAKE_OWNERSHIP);
    ankerl::unordered_dense::map<size_t, CameraModel> cam_models;
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

    addDownwardsPrior({});
}

void RelaxProblem::setupGroundPlaneProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                           ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                                           const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
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

    addDownwardsPrior(options);
    addGPSPositionPrior(graph, options, false);
}

void RelaxProblem::setupGroundMeshProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                          ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                                          const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
                                          const RelaxOptionSet &options,
                                          const std::vector<surface_model> &previousSurfaces, double grid_fraction)
{
    initialize(nodes, cam_models);
    initializeGroundMesh(previousSurfaces, options.get(Option::MINIMAL_MESH));
    _loss.Reset(new ceres::HuberLoss(1 * M_PI / 180), ceres::TAKE_OWNERSHIP);

    // Phase 1: collect track data from all edges (unfiltered)
    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            collectEdgeTracks(graph, edge_id, *edge);
        }
    }

    // Phase 2: build multi-ray tracks, grid-filter by track length, add costs
    addMultiRayTrackCosts(graph, options, grid_fraction);

    // Phase 3: 2-ray fallback for grid cells not covered by multi-ray tracks
    gridFilterMatchesPerImage(graph, edges_to_optimize, grid_fraction);
    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            addRayTriangleMeasurementCost(graph, edge_id, *edge, options);
        }
    }

    addGPSPositionPrior(graph, options, false);
    addMeshFlatPrior();
    addMeshSmoothPrior();
    addMonotonicityCosts();
}

void RelaxProblem::setup3dPointProblem(const MeasurementGraph &graph, std::vector<opencalibration::NodePose> &nodes,
                                       ankerl::unordered_dense::map<size_t, opencalibration::CameraModel> &cam_models,
                                       const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
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

    addGPSPositionPrior(graph, options, true);
    addMonotonicityCosts();

    _solver_options.max_num_iterations = 1000;
    _solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
}

void RelaxProblem::setupTriangulatedRaysProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                                ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                                                const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
                                                const RelaxOptionSet &options)
{
    initialize(nodes, cam_models);
    _loss.Reset(new ceres::HuberLoss(0.2 * M_PI / 180), ceres::TAKE_OWNERSHIP);

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
            collectEdgeTracks(graph, edge_id, *edge);
    }
    addMultiRayTrackCosts(graph, options, 0.05);

    gridFilterMatchesPerImage(graph, edges_to_optimize, 0.05);

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            addTriangulatedRaysCost(graph, edge_id, *edge, options);
        }
    }

    addGPSPositionPrior(graph, options, false);
}

void RelaxProblem::addTriangulatedRaysCost(const MeasurementGraph &graph, size_t edge_id,
                                           const MeasurementGraph::Edge &edge, const RelaxOptionSet &options)
{
    OptimizationPackage pkg;
    pkg.source = nodeid2poseopt(graph, edge.getSource());
    pkg.dest = nodeid2poseopt(graph, edge.getDest());

    if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
        return;

    const auto &source_whitelist = _grid_filter[edge.getSource()][edge_id].getBestMeasurementsPerCell();
    const auto &dest_whitelist = _grid_filter[edge.getDest()][edge_id].getBestMeasurementsPerCell();

    bool points_added = false;
    for (const auto &inlier : edge.payload.inlier_matches)
    {
        if (source_whitelist.find(&inlier) == source_whitelist.end() &&
            dest_whitelist.find(&inlier) == dest_whitelist.end())
            continue;

        if (coveredByMultiRayTracks(edge, inlier, *pkg.source.model_ptr, *pkg.dest.model_ptr))
            continue;

        _problem->AddResidualBlock(
            newAutoDiffTriangulatedReprojectionCost(
                {image_to_3d(inlier.pixel_1, *pkg.source.model_ptr), image_to_3d(inlier.pixel_2, *pkg.dest.model_ptr)},
                !options.hasAll({Option::POSITION})
                    ? std::vector<Eigen::Vector3d>{*pkg.source.loc_ptr, *pkg.dest.loc_ptr}
                    : std::vector<Eigen::Vector3d>{}),
            &_loss, pkg.source.pose_ptr, pkg.dest.pose_ptr);
        points_added = true;
    }

    if (points_added)
    {
        setPoseParameterization(pkg.source.pose_ptr, pkg.source.optimize, options);
        setPoseParameterization(pkg.dest.pose_ptr, pkg.dest.optimize, options);
    }
}

void RelaxProblem::initialize(std::vector<NodePose> &nodes,
                              ankerl::unordered_dense::map<size_t, CameraModel> &cam_models)
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

bool RelaxProblem::shouldAddEdgeToOptimization(const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
                                               size_t edge_id)
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
    if (po.rot_ptr != nullptr)
    {
        po.pose_ptr = poseBlock(node_id, *po.rot_ptr, *po.loc_ptr);
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

double *RelaxProblem::poseBlock(size_t node_id, const Eigen::Quaterniond &orientation,
                                const Eigen::Vector3d &position)
{
    auto [iter, inserted] = _pose_blocks.try_emplace(node_id);
    if (inserted)
    {
        Eigen::Map<Eigen::Quaterniond>(iter->second.data()) = orientation;
        Eigen::Map<Eigen::Vector3d>(iter->second.data() + 4) = position;
    }
    return iter->second.data();
}

void RelaxProblem::setPoseParameterization(double *pose, bool optimize, const RelaxOptionSet &options)
{
    if (_problem->ParameterBlockSize(pose) == 4)
    {
        _problem->SetManifold(pose, &_orientation_parameterization);
    }
    else if (options.hasAll({Option::POSITION}))
    {
        _problem->SetManifold(pose, &_pose_parameterization);
    }
    else
    {
        _problem->SetManifold(pose, &_pose_fixed_position_parameterization);
    }
    if (!optimize)
    {
        _problem->SetParameterBlockConstant(pose);
    }
}

void RelaxProblem::gridFilterMatchesPerImage(const MeasurementGraph &graph,
                                             const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
                                             double grid_cell_image_fraction)
{
    for (size_t edge_id : edges_to_optimize)
    {
        const auto *edge_ptr = graph.getEdge(edge_id);
        if (edge_ptr == nullptr)
            continue;

        const MeasurementGraph::Edge &edge = *edge_ptr;

        OptimizationPackage pkg;
        pkg.relations = &edge.payload;
        pkg.source = nodeid2poseopt(graph, edge.getSource());
        pkg.dest = nodeid2poseopt(graph, edge.getDest());

        if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
            continue;

        const auto &source_model = *graph.getNode(edge.getSource())->payload.model;
        const auto &dest_model = *graph.getNode(edge.getDest())->payload.model;

        Eigen::Matrix3d source_rot = pkg.source.rot_ptr->toRotationMatrix();
        Eigen::Matrix3d dest_rot = pkg.dest.rot_ptr->toRotationMatrix();

        auto &source_filter = _grid_filter[edge.getSource()][edge_id];
        auto &dest_filter = _grid_filter[edge.getDest()][edge_id];
        source_filter.setResolution(grid_cell_image_fraction);
        dest_filter.setResolution(grid_cell_image_fraction);

        std::vector<std::pair<double, size_t>> scored_indices;
        scored_indices.reserve(edge.payload.inlier_matches.size());

        for (size_t idx = 0; idx < edge.payload.inlier_matches.size(); idx++)
        {
            const auto &inlier = edge.payload.inlier_matches[idx];
            ray_d source_ray = {source_rot * image_to_3d(inlier.pixel_1, source_model), *pkg.source.loc_ptr};
            ray_d dest_ray = {dest_rot * image_to_3d(inlier.pixel_2, dest_model), *pkg.dest.loc_ptr};
            const auto intersection = rayIntersection(source_ray, dest_ray);

            const double intersection_score = intersection.second < 0 ? 0. : 1. / (1. + intersection.second);
            const double cos_angle = source_ray.dir.dot(dest_ray.dir);
            const double angle_score = 1.0 - cos_angle * cos_angle;
            const double descriptor_score = inlier.match_index < pkg.relations->matches.size()
                                                ? 1.0 - pkg.relations->matches[inlier.match_index].distance
                                                : 1.0;
            const Eigen::Vector2d src_n =
                (inlier.pixel_1 - source_model.principle_point) / source_model.focal_length_pixels;
            const Eigen::Vector2d dst_n =
                (inlier.pixel_2 - dest_model.principle_point) / dest_model.focal_length_pixels;
            const double ransac_score =
                pkg.relations->relationType == camera_relations::RelationType::HOMOGRAPHY
                    ? 1.0 /
                          (1.0 + (dst_n - (pkg.relations->ransac_relation * src_n.homogeneous()).hnormalized()).norm())
                    : 1.0;

            scored_indices.emplace_back(intersection_score * angle_score * descriptor_score * ransac_score, idx);
        }

        std::sort(scored_indices.begin(), scored_indices.end(),
                  [](const auto &a, const auto &b) { return a.first > b.first; });

        for (const auto &[score, idx] : scored_indices)
        {
            if (score > 0)
            {
                const auto &inlier = edge.payload.inlier_matches[idx];
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

    std::unique_ptr<ceres::CostFunction> func(newAutoDiffMultiDecomposedRotationCost(*pkg.relations));

    double *datas[2] = {pkg.source.pose_ptr, pkg.dest.pose_ptr};

    _problem->AddResidualBlock(func.release(), &_loss, datas[0], datas[1]);
    setPoseParameterization(datas[0], pkg.source.optimize, {Option::ORIENTATION});
    setPoseParameterization(datas[1], pkg.dest.optimize, {Option::ORIENTATION});

    _edges_used.emplace(edge_id);
}

void RelaxProblem::collectEdgeTracks(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge)
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

    for (const auto &inlier : edge.payload.inlier_matches)
    {
        ray_d sourceRay, destRay;
        sourceRay.dir = image_to_3d(inlier.pixel_1, source_model);
        sourceRay.offset = *pkg.source.loc_ptr;
        destRay.dir = image_to_3d(inlier.pixel_2, dest_model);
        destRay.offset = *pkg.dest.loc_ptr;

        auto sourceDestIntersection = rayIntersection(ray_d{*pkg.source.rot_ptr * sourceRay.dir, sourceRay.offset},
                                                      ray_d{*pkg.dest.rot_ptr * destRay.dir, destRay.offset});

        NodeIdFeatureIndex nifi[2];
        nifi[0].node_id = edge.getSource();
        nifi[0].feature_index = inlier.feature_index_1;
        nifi[1].node_id = edge.getDest();
        nifi[1].feature_index = inlier.feature_index_2;
        points.emplace_back(
            FeatureTrack{sourceDestIntersection.first, sourceDestIntersection.second, {nifi[0], nifi[1]}});
    }
}

uint64_t RelaxProblem::trackCellKey(const Eigen::Vector2d &pixel, const CameraModel &model) const
{
    int gi = static_cast<int>(std::floor((pixel.x() / model.pixels_cols) / _track_grid_fraction));
    int gj = static_cast<int>(std::floor((pixel.y() / model.pixels_rows) / _track_grid_fraction));
    return gridCellKey(gi, gj);
}

bool RelaxProblem::coveredByMultiRayTracks(const MeasurementGraph::Edge &edge,
                                           const feature_match_denormalized &inlier, const CameraModel &source_model,
                                           const CameraModel &dest_model) const
{
    if (_multi_ray_measurements.contains(NodeIdFeatureIndex{edge.getSource(), inlier.feature_index_1}) ||
        _multi_ray_measurements.contains(NodeIdFeatureIndex{edge.getDest(), inlier.feature_index_2}))
        return true;

    auto cellCovered = [this](size_t node_id, const Eigen::Vector2d &pixel, const CameraModel &model) {
        auto cells = _multi_ray_covered_cells.find(node_id);
        if (cells == _multi_ray_covered_cells.end())
            return false;
        return cells->second.contains(trackCellKey(pixel, model));
    };
    return cellCovered(edge.getSource(), inlier.pixel_1, source_model) &&
           cellCovered(edge.getDest(), inlier.pixel_2, dest_model);
}

void RelaxProblem::addRayTriangleMeasurementCost(const MeasurementGraph &graph, size_t edge_id,
                                                 const MeasurementGraph::Edge &edge, const RelaxOptionSet &options)
{
    OptimizationPackage pkg;
    pkg.relations = &edge.payload;
    pkg.source = nodeid2poseopt(graph, edge.getSource());
    pkg.dest = nodeid2poseopt(graph, edge.getDest());

    if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
        return;

    const auto &source_model = *pkg.source.model_ptr;
    const auto &dest_model = *pkg.dest.model_ptr;

    auto inverse_iter = _inverse_cam_model_to_optimize.find(source_model.id);
    if (inverse_iter == _inverse_cam_model_to_optimize.end())
    {
        _inverse_cam_model_to_optimize[source_model.id] = convertModel(source_model);
        inverse_iter = _inverse_cam_model_to_optimize.find(source_model.id);
    }

    const auto &source_whitelist = _grid_filter[edge.getSource()][edge_id].getBestMeasurementsPerCell();
    const auto &dest_whitelist = _grid_filter[edge.getDest()][edge_id].getBestMeasurementsPerCell();

    const std::array<double *, 2> datas = {pkg.source.pose_ptr, pkg.dest.pose_ptr};

    MeshIntersectionSearcher intersectionSearcher;
    if (!intersectionSearcher.init(_mesh))
    {
        spdlog::debug("could not initialize mesh searcher, skipping edge");
        return;
    }

    bool points_added = false;
    for (const auto &inlier : edge.payload.inlier_matches)
    {
        if (source_whitelist.find(&inlier) == source_whitelist.end() &&
            dest_whitelist.find(&inlier) == dest_whitelist.end())
            continue;

        if (coveredByMultiRayTracks(edge, inlier, source_model, dest_model))
            continue;

        ray_d sourceRay, destRay;
        sourceRay.dir = image_to_3d(inlier.pixel_1, source_model);
        sourceRay.offset = *pkg.source.loc_ptr;
        destRay.dir = image_to_3d(inlier.pixel_2, dest_model);
        destRay.offset = *pkg.dest.loc_ptr;

        auto sourceDestIntersection = rayIntersection(ray_d{*pkg.source.rot_ptr * sourceRay.dir, sourceRay.offset},
                                                      ray_d{*pkg.dest.rot_ptr * destRay.dir, destRay.offset});

        const double mean_cam_z = (pkg.source.loc_ptr->z() + pkg.dest.loc_ptr->z()) * 0.5;
        const auto intersectionTriangle = intersectionSearcher.triangleIntersect(
            ray_d{Eigen::Vector3d(0, 0, -1),
                  {sourceDestIntersection.first.x(), sourceDestIntersection.first.y(), mean_cam_z}});
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

        if (options.hasAny(
                RelaxOptionSet{Option::FOCAL_LENGTH, Option::PRINCIPAL_POINT, Option::LENS_DISTORTIONS_RADIAL}) &&
            source_model == dest_model)
        {

            std::unique_ptr<ceres::CostFunction> func(newAutoDiffPlaneIntersectionAngleCost_FocalRadial(
                inlier.pixel_1, inlier.pixel_2, corner2d[0], corner2d[1], corner2d[2], inverse_iter->second));

            _problem->AddResidualBlock(func.release(), &_loss, datas[0], datas[1], zValues[0], zValues[1], zValues[2],
                                       &inverse_iter->second.focal_length_pixels,
                                       inverse_iter->second.principle_point.data(),
                                       inverse_iter->second.radial_distortion.data());
            _problem->SetParameterLowerBound(&inverse_iter->second.focal_length_pixels, 0, 100.0);
            _problem->SetParameterUpperBound(&inverse_iter->second.focal_length_pixels, 0, 20000.0);
            if (!options.hasAny(RelaxOptionSet{Option::FOCAL_LENGTH}))
            {
                _problem->SetParameterBlockConstant(&inverse_iter->second.focal_length_pixels);
            }
            if (!options.hasAny(RelaxOptionSet{Option::PRINCIPAL_POINT}))
            {
                _problem->SetParameterBlockConstant(inverse_iter->second.principle_point.data());
            }
            setRadialDistortionParameterization(inverse_iter->second.radial_distortion.data(), options);
            trackRadialObservation(inverse_iter->second.radial_distortion.data(), source_model.pixels_rows,
                                   source_model.pixels_cols, inverse_iter->second.focal_length_pixels);
            points_added = true;
        }
        else
        {
            std::unique_ptr<ceres::CostFunction> func(newAutoDiffPlaneIntersectionAngleCost(
                sourceRay.dir, destRay.dir, corner2d[0], corner2d[1], corner2d[2]));

            _problem->AddResidualBlock(func.release(), &_loss, datas[0], datas[1], zValues[0], zValues[1], zValues[2]);
            points_added = true;
        }
    }
    if (points_added)
    {
        setPoseParameterization(datas[0], pkg.source.optimize, options);
        setPoseParameterization(datas[1], pkg.dest.optimize, options);
    }

    _edges_used.insert(edge_id);
}

void RelaxProblem::addMultiRayTrackCosts(const MeasurementGraph &graph, const RelaxOptionSet &options,
                                         double grid_fraction)
{
    _track_grid_fraction = grid_fraction;
    const bool triangulate = options.get(Option::TRIANGULATED_RAYS);

    // Merge edge tracks into multi-image tracks via UnionFind
    std::vector<const FeatureTrack *> flat_tracks;
    for (const auto &[edge_id, tracks] : _edge_tracks)
    {
        for (const auto &t : tracks)
        {
            flat_tracks.push_back(&t);
        }
    }

    if (flat_tracks.empty())
        return;

    UnionFind uf(flat_tracks.size());
    ankerl::unordered_dense::map<NodeIdFeatureIndex, size_t, NodeIdFeatureIndex> measurement_to_idx;

    for (size_t i = 0; i < flat_tracks.size(); i++)
    {
        for (const auto &m : flat_tracks[i]->measurements)
        {
            auto [it, inserted] = measurement_to_idx.try_emplace(m, i);
            if (!inserted)
            {
                uf.unite(i, it->second);
            }
        }
    }

    // Group measurements by track root
    ankerl::unordered_dense::map<size_t, std::vector<TrackRay>> track_rays;

    for (size_t i = 0; i < flat_tracks.size(); i++)
    {
        size_t root = uf.find(i);
        auto &rays = track_rays[root];

        for (const auto &m : flat_tracks[i]->measurements)
        {
            // Deduplicate by node_id
            bool already_present = false;
            for (const auto &existing : rays) // NOLINT(modernize-loop-convert)
            {
                if (existing.node_id == m.node_id)
                {
                    already_present = true;
                    break;
                }
            }
            if (already_present)
                continue;

            const auto po = nodeid2poseopt(graph, m.node_id, false);
            if (po.pose_ptr == nullptr)
                continue;

            const auto *node = graph.getNode(m.node_id);
            if (node == nullptr || m.feature_index >= node->payload.features.size())
                continue;

            const auto &model = *node->payload.model;
            const auto &pixel = node->payload.features[m.feature_index].location;

            rays.push_back(TrackRay{m.node_id, m.feature_index, model.id, *po.loc_ptr, image_to_3d(pixel, model),
                                   pixel, *po.rot_ptr, po.pose_ptr, po.optimize});
        }
    }

    // Grid-filter tracks by length: keep the longest track per grid cell per image
    ankerl::unordered_dense::map<size_t, GridFilter<size_t>> track_grid_filter;
    for (auto &[root, rays] : track_rays)
    {
        if (rays.size() < 3)
            continue;

        double score = static_cast<double>(rays.size());
        for (const auto &r : rays)
        {
            const auto *node = graph.getNode(r.node_id);
            if (node == nullptr)
                continue;
            const auto &model = *node->payload.model;
            auto &filter = track_grid_filter[r.node_id];
            filter.setResolution(grid_fraction);
            filter.addMeasurement(r.pixel.x() / model.pixels_cols, r.pixel.y() / model.pixels_rows, score, root);
        }
    }

    // Collect the set of track roots that survived filtering
    ankerl::unordered_dense::set<size_t> accepted_tracks;
    for (const auto &[node_id, filter] : track_grid_filter)
    {
        for (size_t root : filter.getBestMeasurementsPerCell())
            accepted_tracks.insert(root);
    }

    MeshIntersectionSearcher intersectionSearcher;
    if (!triangulate && !intersectionSearcher.init(_mesh))
        return;

    size_t tracks_added = 0;
    for (auto &[root, rays] : track_rays)
    {
        if (rays.size() < 3)
            continue;

        if (!accepted_tracks.contains(root))
            continue;

        const std::vector<TrackRay> good_rays =
            triangulate ? addTriangulatedTrackCost(rays, !options.hasAll({Option::POSITION}))
                        : addMeshTrackCost(graph, rays, options, intersectionSearcher);
        if (good_rays.empty())
            continue;

        for (const auto &ray : good_rays)
        {
            setPoseParameterization(ray.pose_ptr, ray.optimize, options);
            _multi_ray_measurements.insert(NodeIdFeatureIndex{ray.node_id, ray.feature_index});

            const auto *node = graph.getNode(ray.node_id);
            if (node != nullptr)
                _multi_ray_covered_cells[ray.node_id].insert(trackCellKey(ray.pixel, *node->payload.model));
        }

        tracks_added++;
    }

    spdlog::info("Added {} multi-ray track costs (3-5 rays)", tracks_added);
}

std::vector<TrackRay> RelaxProblem::addMeshTrackCost(const MeasurementGraph &graph, const std::vector<TrackRay> &rays,
                                                     const RelaxOptionSet &options,
                                                     MeshIntersectionSearcher &intersectionSearcher)
{
    // Compute mean camera location for mesh triangle lookup
    Eigen::Vector3d mean_loc = Eigen::Vector3d::Zero();
    for (const auto &r : rays)
        mean_loc += r.camera_loc;
    mean_loc /= static_cast<double>(rays.size());

    // Intersect first ray pair to find approximate 3D point for mesh lookup
    Eigen::Vector3d ray0_world = rays[0].orientation * rays[0].camera_ray;
    Eigen::Vector3d ray1_world = rays[1].orientation * rays[1].camera_ray;
    auto intersection_3d =
        rayIntersection(ray_d{ray0_world, rays[0].camera_loc}, ray_d{ray1_world, rays[1].camera_loc});

    if (!intersection_3d.first.allFinite())
        return {};

    // Find the mesh triangle at this point
    const auto tri = intersectionSearcher.triangleIntersect(
        ray_d{Eigen::Vector3d(0, 0, -1), {intersection_3d.first.x(), intersection_3d.first.y(), mean_loc.z()}});
    if (tri.type != MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
        return {};

    const auto &triangle = tri.nodeLocations;
    std::array<Eigen::Vector2d, 3> corner2d = {triangle[0]->topRows<2>(), triangle[1]->topRows<2>(),
                                               triangle[2]->topRows<2>()};
    std::array<double *, 3> zValues;
    for (size_t i = 0; i < 3; i++)
        zValues[i] = const_cast<double *>(&triangle[i]->z());

    // Outlier rejection: compute per-ray intersection distance from centroid
    std::vector<std::pair<double, size_t>> ray_scores(rays.size());
    {
        plane_3_corners_d plane3;
        for (int i = 0; i < 3; i++)
            plane3.corner[i] = *triangle[i];
        auto pno = cornerPlane2normOffsetPlane(plane3);

        std::vector<Eigen::Vector3d> intersections(rays.size());
        bool all_valid = true;
        double avg_dist = 0;
        for (size_t i = 0; i < rays.size(); i++)
        {
            ray_d world_ray{rays[i].orientation * rays[i].camera_ray, rays[i].camera_loc};
            all_valid &= rayPlaneIntersection(world_ray, pno, intersections[i]);
            avg_dist += (intersections[i] - rays[i].camera_loc).norm();
        }
        if (!all_valid)
            return {};
        avg_dist /= static_cast<double>(rays.size());

        int n = std::min(static_cast<int>(intersections.size()), ROBUST_CENTROID_MAX_POINTS);
        double huber_threshold = avg_dist * 0.01;
        Eigen::Vector3d centroid = robustCentroid(intersections.data(), n, huber_threshold);

        for (size_t i = 0; i < rays.size(); i++)
        {
            double angle_err = (intersections[i] - centroid).norm() / avg_dist;
            ray_scores[i] = {angle_err, i};
        }
    }

    std::vector<TrackRay> good_rays = selectInlierRays(ray_scores, rays);
    if (good_rays.size() < 3)
        return {};

    const int N = static_cast<int>(good_rays.size());

    // Check if all rays share the same camera model for intrinsics optimization
    bool all_same_model = true;
    for (int i = 1; i < N; i++)
    {
        if (good_rays[i].camera_model_id != good_rays[0].camera_model_id)
        {
            all_same_model = false;
            break;
        }
    }

    bool use_focal_radial =
        all_same_model &&
        options.hasAny(RelaxOptionSet{Option::FOCAL_LENGTH, Option::PRINCIPAL_POINT, Option::LENS_DISTORTIONS_RADIAL});

    std::vector<double *> param_blocks;
    ceres::CostFunction *cost = nullptr;

    InverseDifferentiableCameraModel<double> *inv_model_ptr = nullptr;

    if (use_focal_radial)
    {
        auto inverse_iter = _inverse_cam_model_to_optimize.find(good_rays[0].camera_model_id);
        if (inverse_iter == _inverse_cam_model_to_optimize.end())
        {
            const auto *node = graph.getNode(good_rays[0].node_id);
            _inverse_cam_model_to_optimize[good_rays[0].camera_model_id] = convertModel(*node->payload.model);
            inverse_iter = _inverse_cam_model_to_optimize.find(good_rays[0].camera_model_id);
        }
        inv_model_ptr = &inverse_iter->second;

        // z-heights, focal, principal, radial, then poses
        for (int i = 0; i < 3; i++)
            param_blocks.push_back(zValues[i]);
        param_blocks.push_back(&inv_model_ptr->focal_length_pixels);
        param_blocks.push_back(inv_model_ptr->principle_point.data());
        param_blocks.push_back(inv_model_ptr->radial_distortion.data());
        std::vector<Eigen::Vector2d> pixels;
        for (int i = 0; i < N; i++)
        {
            param_blocks.push_back(good_rays[i].pose_ptr);
            pixels.push_back(good_rays[i].pixel);
        }

        cost = newAutoDiffPlaneIntersectionAngleCost_NRay_FocalRadial(pixels, corner2d, *inv_model_ptr);
    }
    else
    {
        // z-heights, then poses
        for (int i = 0; i < 3; i++)
            param_blocks.push_back(zValues[i]);
        std::vector<Eigen::Vector3d> camera_rays;
        for (int i = 0; i < N; i++)
        {
            param_blocks.push_back(good_rays[i].pose_ptr);
            camera_rays.push_back(good_rays[i].camera_ray);
        }

        cost = newAutoDiffPlaneIntersectionAngleCost_NRay(camera_rays, corner2d);
    }

    if (cost == nullptr)
        return {};

    _problem->AddResidualBlock(cost, nullptr, param_blocks);

    if (inv_model_ptr != nullptr)
    {
        _problem->SetParameterLowerBound(&inv_model_ptr->focal_length_pixels, 0, 100.0);
        _problem->SetParameterUpperBound(&inv_model_ptr->focal_length_pixels, 0, 20000.0);
        if (!options.hasAny(RelaxOptionSet{Option::FOCAL_LENGTH}))
            _problem->SetParameterBlockConstant(&inv_model_ptr->focal_length_pixels);
        if (!options.hasAny(RelaxOptionSet{Option::PRINCIPAL_POINT}))
            _problem->SetParameterBlockConstant(inv_model_ptr->principle_point.data());
        setRadialDistortionParameterization(inv_model_ptr->radial_distortion.data(), options);

        const auto *node = graph.getNode(good_rays[0].node_id);
        trackRadialObservation(inv_model_ptr->radial_distortion.data(), node->payload.model->pixels_rows,
                               node->payload.model->pixels_cols, inv_model_ptr->focal_length_pixels);
    }

    return good_rays;
}

// Reject rays with error > 3x median
std::vector<TrackRay> RelaxProblem::selectInlierRays(std::vector<std::pair<double, size_t>> &ray_scores,
                                                     const std::vector<TrackRay> &rays)
{
    std::sort(ray_scores.begin(), ray_scores.end());
    const double threshold = std::max(ray_scores[ray_scores.size() / 2].first * 3.0, 1e-6);

    std::vector<TrackRay> good_rays;
    for (const auto &[err, idx] : ray_scores)
    {
        if (err <= threshold && good_rays.size() < 5)
            good_rays.push_back(rays[idx]);
    }
    return good_rays;
}

std::vector<TrackRay> RelaxProblem::addTriangulatedTrackCost(const std::vector<TrackRay> &rays, bool fix_positions)
{
    Eigen::Matrix3d normal_matrix = Eigen::Matrix3d::Zero();
    Eigen::Vector3d normal_rhs = Eigen::Vector3d::Zero();
    for (const auto &r : rays)
    {
        const Eigen::Vector3d dir = (r.orientation * r.camera_ray).normalized();
        const Eigen::Matrix3d perpendicular = Eigen::Matrix3d::Identity() - dir * dir.transpose();
        normal_matrix += perpendicular;
        normal_rhs += perpendicular * r.camera_loc;
    }
    const Eigen::Vector3d point = normal_matrix.ldlt().solve(normal_rhs);
    if (!point.allFinite())
        return {};

    std::vector<std::pair<double, size_t>> ray_scores(rays.size());
    for (size_t i = 0; i < rays.size(); i++)
    {
        const Eigen::Vector3d p_cam = rays[i].orientation.inverse() * (point - rays[i].camera_loc);
        ray_scores[i] = {(p_cam.normalized() - rays[i].camera_ray.normalized()).norm(), i};
    }

    const std::vector<TrackRay> good_rays = selectInlierRays(ray_scores, rays);
    if (good_rays.size() < 3 ||
        std::none_of(good_rays.begin(), good_rays.end(), [](const TrackRay &r) { return r.optimize; }))
        return {};

    std::vector<Eigen::Vector3d> camera_rays, camera_positions;
    std::vector<double *> param_blocks;
    for (const auto &r : good_rays)
    {
        camera_rays.push_back(r.camera_ray);
        if (fix_positions)
            camera_positions.push_back(r.camera_loc);
        param_blocks.push_back(r.pose_ptr);
    }
    _problem->AddResidualBlock(newAutoDiffTriangulatedReprojectionCost(camera_rays, camera_positions), &_loss,
                               param_blocks);
    return good_rays;
}

void RelaxProblem::relaxObservedModelOnly()
{
    // optimize just the 3d points to start, since we don't do proper triangulation

    std::vector<double *> params;
    _problem->GetParameterBlocks(&params);

    ankerl::unordered_dense::map<double *, bool> params_map;

    // back up parameters and set everything to constant
    for (double *p : params)
    {
        const bool isConst = _problem->IsParameterBlockConstant(p);
        _problem->SetParameterBlockConstant(p);
        params_map.emplace(p, isConst);
    }

    // set the mesh nodes and 3d points to variable
    for (auto &et : _edge_tracks)
    {
        for (auto &t : et.second)
        {
            auto param = params_map.find(t.point.data());
            if (param != params_map.end() && !param->second)
            {
                _problem->SetParameterBlockVariable(t.point.data());
            }
        }
    }
    for (auto iter = _mesh.nodebegin(); iter != _mesh.nodeend(); ++iter)
    {
        auto &p = iter->second.payload.location;
        auto param = params_map.find(&p.z());
        if (param != params_map.end() && !param->second)
            _problem->SetParameterBlockVariable(&p.z());
    }

    // solve
    spdlog::debug("optimizing surface only");
    solve();

    // restore parameter const-ness from backup
    for (const auto &[param, isConst] : params_map)
    {
        if (isConst)
        {
            _problem->SetParameterBlockConstant(param);
        }
        else
        {
            _problem->SetParameterBlockVariable(param);
        }
    }
}

void RelaxProblem::addPointMeasurementsCost(const MeasurementGraph &graph, size_t edge_id,
                                            const MeasurementGraph::Edge &edge, const RelaxOptionSet &options)
{
    if (!options.hasAll({Option::ORIENTATION, Option::POINTS_3D}))
    {
        spdlog::critical("No viable bundle options found");
        return;
    }

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

    double *pose_ptrs[2] = {pkg.source.pose_ptr, pkg.dest.pose_ptr};
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
        double *principals[2] = {source_model.principle_point.data(), dest_model.principle_point.data()};
        double *radials[2] = {source_model.radial_distortion.data(), dest_model.radial_distortion.data()};
        double *tangentials[2] = {source_model.tangential_distortion.data(), dest_model.tangential_distortion.data()};

        if (options.hasAny({Option::LENS_DISTORTIONS_TANGENTIAL}) &&
            options.hasAll(
                {Option::LENS_DISTORTIONS_RADIAL, Option::FOCAL_LENGTH, Option::ORIENTATION, Option::POINTS_3D}))
        {
            func[0].reset(newAutoDiffPixelErrorCost_OrientationFocalRadialTangential(source_model,
                                                                                     inlier.pixel_1));
            func[1].reset(newAutoDiffPixelErrorCost_OrientationFocalRadialTangential(dest_model,
                                                                                     inlier.pixel_2));
            for (int i = 0; i < 2; i++)
            {
                args[i] = {pose_ptrs[i], points.back().point.data(), focals[i], principals[i], radials[i],
                           tangentials[i]};
            }
        }
        else if (options.hasAny({Option::LENS_DISTORTIONS_RADIAL}) &&
                 options.hasAll({Option::FOCAL_LENGTH, Option::ORIENTATION, Option::POINTS_3D}))
        {
            func[0].reset(
                newAutoDiffPixelErrorCost_OrientationFocalRadial(source_model, inlier.pixel_1));
            func[1].reset(
                newAutoDiffPixelErrorCost_OrientationFocalRadial(dest_model, inlier.pixel_2));
            for (int i = 0; i < 2; i++)
            {
                args[i] = {pose_ptrs[i], points.back().point.data(), focals[i], principals[i], radials[i]};
            }
        }
        else if (options.hasAny({Option::FOCAL_LENGTH, Option::PRINCIPAL_POINT}) &&
                 options.hasAll({Option::ORIENTATION, Option::POINTS_3D}))
        {
            func[0].reset(
                newAutoDiffPixelErrorCost_OrientationFocal(source_model, inlier.pixel_1));
            func[1].reset(newAutoDiffPixelErrorCost_OrientationFocal(dest_model, inlier.pixel_2));

            for (int i = 0; i < 2; i++)
            {
                args[i] = {pose_ptrs[i], points.back().point.data(), focals[i], principals[i]};
            }
        }
        else
        {
            func[0].reset(newAutoDiffPixelErrorCost_Orientation(source_model, inlier.pixel_1));
            func[1].reset(newAutoDiffPixelErrorCost_Orientation(dest_model, inlier.pixel_2));

            for (int i = 0; i < 2; i++)
            {
                args[i] = {pose_ptrs[i], points.back().point.data()};
            }
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

        if (options.hasAny({Option::LENS_DISTORTIONS_RADIAL}))
        {
            for (int i = 0; i < 2; i++)
            {
                const auto &m = (i == 0) ? source_model : dest_model;
                trackRadialObservation(radials[i], m.pixels_rows, m.pixels_cols, m.focal_length_pixels);
            }
        }

        points_added = true;
    }

    if (points_added)
    {
        setPoseParameterization(pose_ptrs[0], pkg.source.optimize, options);
        setPoseParameterization(pose_ptrs[1], pkg.dest.optimize, options);

        if (options.hasAny({Option::FOCAL_LENGTH, Option::PRINCIPAL_POINT, Option::LENS_DISTORTIONS_RADIAL,
                            Option::LENS_DISTORTIONS_TANGENTIAL}))
        {
            if (!options.hasAny({Option::FOCAL_LENGTH}))
            {
                _problem->SetParameterBlockConstant(&source_model.focal_length_pixels);
                _problem->SetParameterBlockConstant(&dest_model.focal_length_pixels);
            }
            else
            {
                _problem->SetParameterLowerBound(&source_model.focal_length_pixels, 0, 100.0);
                _problem->SetParameterLowerBound(&dest_model.focal_length_pixels, 0, 100.0);
                _problem->SetParameterUpperBound(&source_model.focal_length_pixels, 0, 20000.0);
                _problem->SetParameterUpperBound(&dest_model.focal_length_pixels, 0, 20000.0);
            }
            if (!options.hasAny({Option::PRINCIPAL_POINT}))
            {
                _problem->SetParameterBlockConstant(source_model.principle_point.data());
                _problem->SetParameterBlockConstant(dest_model.principle_point.data());
            }
        }

        if (options.hasAll({Option::LENS_DISTORTIONS_RADIAL}))
        {
            if (options.hasAll({Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION}))
            {
                _problem->SetManifold(source_model.radial_distortion.data(), &_brown246_parameterization);
                _problem->SetManifold(dest_model.radial_distortion.data(), &_brown246_parameterization);
            }
            else if (options.hasAll({Option::LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION}))
            {
                _problem->SetManifold(source_model.radial_distortion.data(), &_brown24_parameterization);
                _problem->SetManifold(dest_model.radial_distortion.data(), &_brown24_parameterization);
            }
            else if (options.hasAll({Option::LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION}))
            {
                _problem->SetManifold(source_model.radial_distortion.data(), &_brown2_parameterization);
                _problem->SetManifold(dest_model.radial_distortion.data(), &_brown2_parameterization);
            }
            else
            {
                spdlog::warn("No parameterization chosen for radial distortion");
            }
        }
    }

    _edges_used.emplace(edge_id);
}

void RelaxProblem::setRadialDistortionParameterization(double *radial_distortion, const RelaxOptionSet &options)
{
    if (!options.hasAny({Option::LENS_DISTORTIONS_RADIAL}))
    {
        _problem->SetParameterBlockConstant(radial_distortion);
    }
    else if (options.hasAll({Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION}))
    {
        _problem->SetManifold(radial_distortion, &_brown246_parameterization);
    }
    else if (options.hasAll({Option::LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION}))
    {
        _problem->SetManifold(radial_distortion, &_brown24_parameterization);
    }
    else if (options.hasAll({Option::LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION}))
    {
        _problem->SetManifold(radial_distortion, &_brown2_parameterization);
    }
    else
    {
        spdlog::warn("No parameterization chosen for radial distortion");
    }
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

void RelaxProblem::initializeGroundMesh(const std::vector<surface_model> &previousSurfaces, bool useMinimalMesh)
{
    point_cloud cameraLocations;
    cameraLocations.reserve(_nodes_to_optimize.size());
    for (const auto &[key, value] : _nodes_to_optimize)
    {
        cameraLocations.push_back(value->position);
    }

    // Check if we have a previous mesh to reuse
    const MeshGraph *previousMesh = nullptr;
    for (const auto &s : previousSurfaces)
    {
        if (s.mesh.size_nodes() > 0)
        {
            previousMesh = &s.mesh;
            break;
        }
    }

    // Don't reuse if:
    // - useMinimalMesh is requested AND previous mesh is just a 3-node triangle (from GROUND_PLANE)
    // In that case, we want to build a proper 4-node minimal mesh instead
    const bool previousIsGroundPlaneTriangle = previousMesh != nullptr && previousMesh->size_nodes() == 3;
    const bool shouldReusePreviousMesh = previousMesh != nullptr && !(useMinimalMesh && previousIsGroundPlaneTriangle);

    if (shouldReusePreviousMesh)
    {
        // Reuse the previous mesh structure (preserves refinement)
        _mesh = *previousMesh;
        spdlog::info("Reusing previous mesh with {} nodes, {} edges", _mesh.size_nodes(), _mesh.size_edges());
    }
    else if (useMinimalMesh)
    {
        // Build minimal 2-triangle mesh for first iteration
        _mesh = buildMinimalMesh(cameraLocations, previousSurfaces);
        spdlog::info("Using minimal 2-triangle mesh with {} nodes, {} edges", _mesh.size_nodes(), _mesh.size_edges());
    }
    else
    {
        // Build grid mesh (legacy behavior)
        _mesh = rebuildMesh(cameraLocations, previousSurfaces);
        spdlog::info("Built grid mesh with {} nodes, {} edges", _mesh.size_nodes(), _mesh.size_edges());
    }
}

void RelaxProblem::addDownwardsPrior(const RelaxOptionSet &options)
{
    for (auto &p : _nodes_to_optimize)
    {
        if (!p.second->orientation.coeffs().hasNaN() && !p.second->position.hasNaN())
        {
            double *d = poseBlock(p.first, p.second->orientation, p.second->position);
            _problem->AddResidualBlock(newAutoDiffPointsDownwardsPrior(1e-3), nullptr, d);
            setPoseParameterization(d, true, options);
        }
    }
}

void RelaxProblem::addGPSPositionPrior(const MeasurementGraph &graph, const RelaxOptionSet &options,
                                       bool pixel_residuals)
{
    if (!options.hasAll({Option::POSITION}))
        return;

    for (const auto &[node_id, pose] : _nodes_to_optimize)
    {
        auto block = _pose_blocks.find(node_id);
        if (block == _pose_blocks.end() || !_problem->HasParameterBlock(block->second.data()))
            continue;
        double *pose_block = block->second.data();

        const auto *node = graph.getNode(node_id);
        if (node == nullptr || node->payload.gps_position.hasNaN())
            continue;

        std::vector<ceres::ResidualBlockId> image_measurements;
        _problem->GetResidualBlocksForParameterBlock(pose_block, &image_measurements);

        const double one_pixel = pixel_residuals ? 1.0 : 1.0 / node->payload.model->focal_length_pixels;
        const double one_pixel_rms = one_pixel * std::sqrt(static_cast<double>(image_measurements.size()));
        _problem->AddResidualBlock(newAutoDiffGPSPositionPrior(node->payload.gps_position,
                                                               one_pixel_rms / GPS_HORIZONTAL_SIGMA_METERS,
                                                               one_pixel_rms / GPS_VERTICAL_SIGMA_METERS),
                                   nullptr, pose_block);
    }
}

void RelaxProblem::addMeshFlatPrior()
{
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

    // Anchor to initial z to prevent gauge freedom drift
    _mesh_initial_z.clear();
    _mesh_initial_z.reserve(_mesh.size_nodes());
    for (auto iter = _mesh.nodebegin(); iter != _mesh.nodeend(); ++iter)
    {
        _mesh_initial_z.push_back(iter->second.payload.location.z());
    }
    size_t i = 0;
    for (auto iter = _mesh.nodebegin(); iter != _mesh.nodeend(); ++iter, ++i)
    {
        double *h = &iter->second.payload.location.z();
        _problem->AddResidualBlock(newAutoDiffDifferenceCost(1e-5), nullptr, h, &_mesh_initial_z[i]);
        _problem->SetParameterBlockConstant(&_mesh_initial_z[i]);
    }
}

void RelaxProblem::addMeshSmoothPrior()
{
    for (auto iter = _mesh.edgebegin(); iter != _mesh.edgeend(); ++iter)
    {
        const MeshEdge &edge = iter->second.payload;
        if (edge.border)
            continue;

        const size_t idA = iter->second.getSource();
        const size_t idB = iter->second.getDest();
        const size_t idC = edge.triangleOppositeNodes[0];
        const size_t idD = edge.triangleOppositeNodes[1];

        auto *nodeA = _mesh.getNode(idA);
        auto *nodeB = _mesh.getNode(idB);
        auto *nodeC = _mesh.getNode(idC);
        auto *nodeD = _mesh.getNode(idD);

        const Eigen::Vector2d xyA = nodeA->payload.location.head<2>();
        const Eigen::Vector2d xyB = nodeB->payload.location.head<2>();
        const Eigen::Vector2d xyC = nodeC->payload.location.head<2>();
        const Eigen::Vector2d xyD = nodeD->payload.location.head<2>();

        double *zA = &nodeA->payload.location.z();
        double *zB = &nodeB->payload.location.z();
        double *zC = &nodeC->payload.location.z();
        double *zD = &nodeD->payload.location.z();

        _problem->AddResidualBlock(newAutoDiffAdjacentTriangleNormalCost(xyA, xyB, xyC, xyD, 1e-4), nullptr, zA, zB, zC,
                                   zD);
    }
}

void RelaxProblem::trackRadialObservation(double *radial_data, size_t pixels_rows, size_t pixels_cols,
                                          double focal_length)
{
    auto &info = _radial_monotonicity_info[radial_data];
    info.observation_count++;
    if (info.observation_count == 1)
    {
        double half_cols = pixels_cols / 2.0;
        double half_rows = pixels_rows / 2.0;
        info.r_max = std::sqrt(half_cols * half_cols + half_rows * half_rows) / focal_length;
    }
}

void RelaxProblem::addMonotonicityCosts()
{
    for (auto &[radial_data, info] : _radial_monotonicity_info)
    {
        double weight = std::sqrt(info.observation_count / 10.0);
        _problem->AddResidualBlock(newAutoDiffDistortionMonotonicityCost(info.r_max, weight), nullptr, radial_data);
    }
}

void RelaxProblem::solve()
{
    std::ostringstream thread_stream;
    thread_stream << std::this_thread::get_id();
    spdlog::info("Thread {} start relax: {} parameter blocks, {} residual blocks", thread_stream.str(),
                 _problem->NumParameterBlocks(), _problem->NumResidualBlocks());

    // Early exit for empty problems (nothing to optimize)
    if (_problem->NumParameterBlocks() == 0 || _problem->NumResidualBlocks() == 0)
    {
        spdlog::info("Thread {} end relax: iterations 0, cost ratio -nan, time {}s", thread_stream.str(), 0.0f);
        return;
    }

    _solver.Solve(_solver_options, _problem.get(), &_summary);
    spdlog::info("Thread {} end relax: iterations {}, cost ratio {}, time {}s", thread_stream.str(),
                 _summary.iterations.size(), static_cast<float>(_summary.final_cost / _summary.initial_cost),
                 static_cast<float>(_summary.total_time_in_seconds));
    spdlog::debug(_summary.FullReport());

    for (auto &p : _nodes_to_optimize)
    {
        auto block = _pose_blocks.find(p.first);
        if (block != _pose_blocks.end())
        {
            p.second->orientation = Eigen::Map<const Eigen::Quaterniond>(block->second.data());
            p.second->position = Eigen::Map<const Eigen::Vector3d>(block->second.data() + 4);
        }
        p.second->orientation.normalize();
    }

    // hackity hackity hack - copy back camera models
    for (const auto &[id, inverse_model] : _inverse_cam_model_to_optimize)
    {
        *_cam_models_to_optimize[id] = CameraModel(convertModel(inverse_model), id);
    }
}

surface_model RelaxProblem::getSurfaceModel()
{
    surface_model s;

    std::vector<const FeatureTrack *> flat_tracks;
    for (const auto &[edge_id, tracks] : _edge_tracks)
    {
        for (const auto &t : tracks)
        {
            flat_tracks.push_back(&t);
        }
    }

    UnionFind uf(flat_tracks.size());
    ankerl::unordered_dense::map<NodeIdFeatureIndex, size_t, NodeIdFeatureIndex> measurement_to_idx;

    for (size_t i = 0; i < flat_tracks.size(); i++)
    {
        const auto &t = *flat_tracks[i];
        if (!t.point.allFinite())
            continue;

        for (const auto &m : t.measurements)
        {
            auto [it, inserted] = measurement_to_idx.try_emplace(m, i);
            if (!inserted)
            {
                uf.unite(i, it->second);
            }
        }
    }

    struct MergedTrack
    {
        std::vector<Eigen::Vector3d> points;
        double min_error = std::numeric_limits<double>::infinity();
        ankerl::unordered_dense::set<size_t> unique_nodes;
    };

    ankerl::unordered_dense::map<size_t, MergedTrack> merged;
    for (size_t i = 0; i < flat_tracks.size(); i++)
    {
        const auto &t = *flat_tracks[i];
        if (!t.point.allFinite())
            continue;

        size_t root = uf.find(i);
        auto &m = merged[root];
        m.points.push_back(t.point);
        if (std::isfinite(t.error))
            m.min_error = std::min(m.min_error, t.error);
        for (const auto &meas : t.measurements)
            m.unique_nodes.insert(meas.node_id);
    }

    point_cloud cloud_points;
    cloud_points.reserve(merged.size());
    for (const auto &[root, m] : merged)
    {
        size_t num_views = m.unique_nodes.size();
        double max_allowed_error = (num_views >= 3) ? 10.0 : 1.0;
        if (m.min_error > max_allowed_error)
            continue;

        Eigen::Vector3d pt;
        if (m.points.size() == 1)
        {
            pt = m.points[0];
        }
        else
        {
            int n = std::min(static_cast<int>(m.points.size()), ROBUST_CENTROID_MAX_POINTS);
            pt = robustCentroid(m.points.data(), n, 1.0);
        }
        cloud_points.push_back(pt);
    }

    if (!cloud_points.empty())
    {
        s.cloud.emplace_back(std::move(cloud_points));
    }

    s.mesh = _mesh;

    return s;
}

} // namespace opencalibration
