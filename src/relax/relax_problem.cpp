#include <opencalibration/relax/relax_problem.hpp>

#include "ceres_log_forwarding.hpp"
#include <opencalibration/relax/autodiff_cost_function.hpp>

namespace opencalibration
{

RelaxProblem::RelaxProblem()
    : _log_forwarder_dependency(GetCeresLogForwarder()), _loss(new ceres::TrivialLoss(), ceres::TAKE_OWNERSHIP)
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
    initialize(nodes);
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
                                           const std::unordered_set<size_t> &edges_to_optimize)
{
    initialize(nodes);
    initializeGroundPlane();
    _loss.Reset(new ceres::HuberLoss(1 * M_PI / 180), ceres::TAKE_OWNERSHIP);
    gridFilterMatchesPerImage(graph, edges_to_optimize);

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            addGlobalPlaneMeasurementsCost(graph, edge_id, *edge);
        }
    }
}

void RelaxProblem::setup3dPointProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                       const std::unordered_set<size_t> &edges_to_optimize)
{
    initialize(nodes);
    _loss.Reset(new ceres::HuberLoss(10), ceres::TAKE_OWNERSHIP);

    gridFilterMatchesPerImage(graph, edges_to_optimize);

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            addPointMeasurementsCost(graph, edge_id, *edge);
        }
    }

    _solver_options.max_num_iterations = 300;
    _solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
}

void RelaxProblem::setup3dTracksProblem(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                        const std::unordered_set<size_t> &edges_to_optimize)
{
    initialize(nodes);
    _loss.Reset(new ceres::HuberLoss(10), ceres::TAKE_OWNERSHIP);

    for (size_t edge_id : edges_to_optimize)
    {
        const MeasurementGraph::Edge *edge = graph.getEdge(edge_id);
        if (edge != nullptr && shouldAddEdgeToOptimization(edges_to_optimize, edge_id))
        {
            insertEdgeTracks(graph, edge_id, *edge);
        }
    }
    compileEdgeTracks(graph);
    filterTracks(graph);
    addTrackCosts(graph);

    _solver_options.max_num_iterations = 300;
    _solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
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

void RelaxProblem::gridFilterMatchesPerImage(const MeasurementGraph &graph,
                                             const std::unordered_set<size_t> &edges_to_optimize)
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

        const auto &source_model = graph.getNode(edge.getSource())->payload.model;
        const auto &dest_model = graph.getNode(edge.getDest())->payload.model;

        Eigen::Matrix3d source_rot = pkg.source.rot_ptr->toRotationMatrix();
        Eigen::Matrix3d dest_rot = pkg.dest.rot_ptr->toRotationMatrix();

        auto &source_filter = _grid_filter[edge.getSource()];
        auto &dest_filter = _grid_filter[edge.getDest()];

        for (const auto &inlier : edge.payload.inlier_matches)
        {
            // make 3D intersection, add it to `points`
            Eigen::Vector3d source_ray = source_rot * image_to_3d(inlier.pixel_1, source_model);
            Eigen::Vector3d dest_ray = dest_rot * image_to_3d(inlier.pixel_2, dest_model);
            auto intersection =
                rayIntersection(ray_d{source_ray, *pkg.source.loc_ptr}, ray_d{dest_ray, *pkg.dest.loc_ptr});

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

void RelaxProblem::addGlobalPlaneMeasurementsCost(const MeasurementGraph &graph, size_t edge_id,
                                                  const MeasurementGraph::Edge &edge)
{
    auto &points = _edge_tracks[edge_id];
    points.reserve(edge.payload.inlier_matches.size());

    OptimizationPackage pkg;
    pkg.relations = &edge.payload;
    pkg.source = nodeid2poseopt(graph, edge.getSource());
    pkg.dest = nodeid2poseopt(graph, edge.getDest());

    if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
        return;

    const auto &source_model = graph.getNode(edge.getSource())->payload.model;
    const auto &dest_model = graph.getNode(edge.getDest())->payload.model;

    auto source_whitelist = _grid_filter[edge.getSource()].getBestMeasurementsPerCell();
    auto dest_whitelist = _grid_filter[edge.getDest()].getBestMeasurementsPerCell();

    double *datas[5] = {pkg.source.rot_ptr->coeffs().data(), pkg.dest.rot_ptr->coeffs().data(),
                        &_global_plane.corner[0].z(), &_global_plane.corner[1].z(), &_global_plane.corner[2].z()};
    Eigen::Vector2d corner2d[3]{_global_plane.corner[0].topRows<2>(), _global_plane.corner[1].topRows<2>(),
                                _global_plane.corner[2].topRows<2>()};
    bool points_added = false;
    for (const auto &inlier : edge.payload.inlier_matches)
    {
        if (source_whitelist.find(&inlier) == source_whitelist.end() &&
            dest_whitelist.find(&inlier) == dest_whitelist.end())
            continue;

        // make 3D intersection, add it to `points`
        Eigen::Vector3d source_ray = image_to_3d(inlier.pixel_1, source_model);
        Eigen::Vector3d dest_ray = image_to_3d(inlier.pixel_2, dest_model);

        // add cost functions for this 3D point from both the source and dest camera
        std::unique_ptr<ceres::CostFunction> func(newAutoDiffPlaneIntersectionAngleCost(
            *pkg.source.loc_ptr, *pkg.dest.loc_ptr, source_ray, dest_ray, corner2d[0], corner2d[1], corner2d[2]));

        _problem->AddResidualBlock(func.release(), &_loss, datas[0], datas[1], datas[2], datas[3], datas[4]);
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

void RelaxProblem::insertEdgeTracks(const MeasurementGraph &graph, size_t edge_id, const MeasurementGraph::Edge &edge)
{
    auto &points = _edge_tracks[edge_id];
    points.reserve(edge.payload.inlier_matches.size());

    OptimizationPackage pkg;
    pkg.relations = &edge.payload;
    pkg.source = nodeid2poseopt(graph, edge.getSource());
    pkg.dest = nodeid2poseopt(graph, edge.getDest());

    if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
        return;

    for (size_t i = 0; i < edge.payload.inlier_matches.size(); i++)
    {
        const auto &inlier = edge.payload.inlier_matches[i];

        auto map_index_1 = NodeIdFeatureIndex{edge.getSource(), inlier.feature_index_1};
        auto map_index_2 = NodeIdFeatureIndex{edge.getDest(), inlier.feature_index_2};

        _node_id_feature_index_tracklinks[map_index_1].emplace_back(map_index_2);
        _node_id_feature_index_tracklinks[map_index_2].emplace_back(map_index_1);
    }
    _edges_used.insert(edge_id);
}

void RelaxProblem::compileEdgeTracks(const MeasurementGraph &graph)
{
    // start quick and dirty, just make pairs of measurements
    using key_t = std::pair<NodeIdFeatureIndex, NodeIdFeatureIndex>;

    struct key_t_hash
    {
        size_t operator()(const key_t &key) const
        {
            std::hash<NodeIdFeatureIndex> h;
            return h(key.first) ^ (31 * h(key.second));
        }
    };

    std::unordered_set<key_t, key_t_hash> used_pairs;
    for (const auto &nifi_links : _node_id_feature_index_tracklinks)
    {
        for (const auto &nifi2 : nifi_links.second)
        {
            key_t key{nifi_links.first, nifi2};
            if (nifi2 < nifi_links.first)
            {
                std::swap(key.first, key.second);
            }

            if (used_pairs.find(key) != used_pairs.end())
            {
                continue;
            }
            used_pairs.insert(key);

            FeatureTrack track;
            track.measurements.reserve(2);
            track.measurements.push_back(nifi_links.first);
            track.measurements.push_back(nifi2);
            _tracks.emplace_back(std::move(track));
        }
    }

    for (auto &track : _tracks)
    {
        std::vector<ray_d> rays;
        for (const auto &nifi : track.measurements)
        {
            OptimizationPackage::PoseOpt po = nodeid2poseopt(graph, nifi.node_id);
            if (po.loc_ptr == nullptr)
                continue;

            const auto *node = graph.getNode(nifi.node_id);
            const auto &image = node->payload;
            Eigen::Vector2d pixel = image.features[nifi.feature_index].location;
            Eigen::Vector3d ray_dir = (*po.rot_ptr) * image_to_3d(pixel, image.model);
            rays.emplace_back(ray_d{ray_dir, *po.loc_ptr});
        }
        std::pair<Eigen::Vector3d, double> point_error = rayIntersection(rays);
        track.point = point_error.first;
        track.error = point_error.second;
    }

    // TODO: merge tracks that have shared features *and* close together triangulated 3d points
}

void RelaxProblem::filterTracks(const MeasurementGraph &graph)
{
    // normalize track error between 0 and 1 so that we can use it to tie track length scores
    double max_track_error = 1e-9, min_track_error = 1e9;
    for (const auto &track : _tracks)
    {
        if (std::abs(track.error) > max_track_error)
            max_track_error = track.error;
        if (std::abs(track.error) < min_track_error)
            min_track_error = track.error;
    }

    // apply grid filter to mark tracks as included or not. One valid measurement required to include a track
    std::unordered_map<size_t, GridFilter<size_t>> node_id_track_index_grid_filter;
    for (size_t i = 0; i < _tracks.size(); i++)
    {
        auto &track = _tracks[i];
        double score =
            track.measurements.size() -
            std::clamp((std::abs(track.error) - min_track_error) / (max_track_error - min_track_error), 0., 0.9999);
        for (const auto &nifi : track.measurements)
        {
            const auto &image = graph.getNode(nifi.node_id)->payload;
            const auto &loc = image.features[nifi.feature_index].location;

            node_id_track_index_grid_filter[nifi.node_id].addMeasurement(loc.x() / image.model.pixels_cols,
                                                                         loc.y() / image.model.pixels_rows, score, i);
        }
    }

    std::vector<size_t> track_scores(_tracks.size(), 0);
    for (const auto &filter : node_id_track_index_grid_filter)
    {
        for (size_t track_id : filter.second.getBestMeasurementsPerCell())
        {
            track_scores[track_id]++;
        }
    }

    const size_t KEEP_THRESHOLD = 1;

    track_vec tracks_to_optimize;
    tracks_to_optimize.reserve(_tracks.size());

    for (size_t i = 0; i < _tracks.size(); i++)
    {
        auto &track = _tracks[i];
        if (track_scores[i] >= KEEP_THRESHOLD && std::isfinite(track.error))
        {
            tracks_to_optimize.emplace_back(std::move(track));
        }
    }
    std::swap(tracks_to_optimize, _tracks);
}

void RelaxProblem::addTrackCosts(const MeasurementGraph &graph)
{
    for (auto &track : _tracks)
    {
        for (const auto &nifi : track.measurements)
        {
            OptimizationPackage::PoseOpt po = nodeid2poseopt(graph, nifi.node_id);
            if (po.loc_ptr == nullptr)
                continue;

            const auto &image = graph.getNode(nifi.node_id)->payload;
            double *data = po.rot_ptr->coeffs().data();

            std::unique_ptr<ceres::CostFunction> func(
                newAutoDiffPixelErrorCost(*po.loc_ptr, image.model, image.features[nifi.feature_index].location));

            Eigen::Vector2d res{NAN, NAN};
            const double *args[2] = {data, track.point.data()};
            func->Evaluate(args, res.data(), nullptr);

            if (!res.array().allFinite())
            {
                spdlog::trace("Skipping adding NaN track measurement residual");
                continue;
            }

            _problem->AddResidualBlock(func.release(), &_loss, data, track.point.data());
            _problem->SetParameterization(data, &_quat_parameterization);

            if (!po.optimize)
            {
                _problem->SetParameterBlockConstant(data);
            }
        }
    }
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
    for (auto &t : _tracks)
    {
        if (params_set.find(t.point.data()) != params_set.end())
            _problem->SetParameterBlockVariable(t.point.data());
    }
    for (auto &et : _edge_tracks)
        for (auto &t : et.second)
            if (params_set.find(t.point.data()) != params_set.end())
                _problem->SetParameterBlockVariable(t.point.data());

    for (auto &p : _global_plane.corner)
        if (params_set.find(&p.z()) != params_set.end())
            _problem->SetParameterBlockVariable(&p.z());

    spdlog::debug("optimizing 3d points only");
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
                                            const MeasurementGraph::Edge &edge)
{
    auto &points = _edge_tracks[edge_id];
    points.reserve(edge.payload.inlier_matches.size());

    OptimizationPackage pkg;
    pkg.relations = &edge.payload;
    pkg.source = nodeid2poseopt(graph, edge.getSource());
    pkg.dest = nodeid2poseopt(graph, edge.getDest());

    if (pkg.source.loc_ptr == nullptr || pkg.dest.loc_ptr == nullptr)
        return;

    const auto &source_model = graph.getNode(edge.getSource())->payload.model;
    const auto &dest_model = graph.getNode(edge.getDest())->payload.model;

    auto source_whitelist = _grid_filter[edge.getSource()].getBestMeasurementsPerCell();
    auto dest_whitelist = _grid_filter[edge.getDest()].getBestMeasurementsPerCell();

    Eigen::Matrix3d source_rot = pkg.source.rot_ptr->toRotationMatrix();
    Eigen::Matrix3d dest_rot = pkg.dest.rot_ptr->toRotationMatrix();

    double *datas[2] = {pkg.source.rot_ptr->coeffs().data(), pkg.dest.rot_ptr->coeffs().data()};
    bool points_added = false;
    for (const auto &inlier : edge.payload.inlier_matches)
    {

        if (source_whitelist.find(&inlier) == source_whitelist.end() &&
            dest_whitelist.find(&inlier) == dest_whitelist.end())
            continue;

        // make 3D intersection, add it to `points`
        Eigen::Vector3d source_ray = source_rot * image_to_3d(inlier.pixel_1, source_model);
        Eigen::Vector3d dest_ray = dest_rot * image_to_3d(inlier.pixel_2, dest_model);
        auto intersection = rayIntersection(ray_d{source_ray, *pkg.source.loc_ptr}, ray_d{dest_ray, *pkg.dest.loc_ptr});
        NodeIdFeatureIndex nifi[2];
        nifi[0].node_id = edge.getSource();
        nifi[0].feature_index = inlier.feature_index_1;
        nifi[1].node_id = edge.getDest();
        nifi[1].feature_index = inlier.feature_index_2;
        points.emplace_back(FeatureTrack{intersection.first, intersection.second, {nifi[0], nifi[1]}});

        // add cost functions for this 3D point from both the source and dest camera
        std::unique_ptr<ceres::CostFunction> func[2];
        func[0].reset(newAutoDiffPixelErrorCost(*pkg.source.loc_ptr, source_model, inlier.pixel_1));
        func[1].reset(newAutoDiffPixelErrorCost(*pkg.dest.loc_ptr, dest_model, inlier.pixel_2));

        bool all_finite = true;
        for (int i = 0; i < 2; i++)
        {
            Eigen::Vector2d res{NAN, NAN};
            const double *args[2] = {datas[i], points.back().point.data()};
            func[0]->Evaluate(args, res.data(), nullptr);

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
            _problem->AddResidualBlock(func[i].release(), &_loss, datas[i], points.back().point.data());
        }
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

    // TODO: use a robust RANSAC based plane fit of the ray-ray intersections instead
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
        _problem->AddResidualBlock(newAutoDiffPointsDownwardsPrior(), nullptr, d);
        _problem->SetParameterization(d, &_quat_parameterization);
    }
}

void RelaxProblem::solve()
{
    spdlog::debug("Start rotation relax: {} active nodes, {} edges", _nodes_to_optimize.size(), _edges_used.size());

    _solver.Solve(_solver_options, _problem.get(), &_summary);
    spdlog::info(_summary.BriefReport());
    spdlog::debug(_summary.FullReport());

    for (auto &p : _nodes_to_optimize)
    {
        p.second->orientation.normalize();
    }
}

} // namespace opencalibration
