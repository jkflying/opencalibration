#include <opencalibration/relax/relax.hpp>

#include <opencalibration/performance/performance.hpp>
#include <opencalibration/relax/relax_problem.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <optional>

namespace
{

using namespace opencalibration;

static const Eigen::Quaterniond DOWN_ORIENTED_NORTH(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));

bool withinNadirCone(const Eigen::Quaterniond &orientation)
{
    constexpr double NADIR_CONE_HALF_ANGLE = M_PI / 4;
    return (orientation * Eigen::Vector3d::UnitZ()).z() < -std::cos(NADIR_CONE_HALF_ANGLE);
}

double tiltDegrees(const Eigen::Quaterniond &orientation)
{
    return std::acos(std::clamp(-(orientation * Eigen::Vector3d::UnitZ()).z(), -1.0, 1.0)) * 180 / M_PI;
}

surface_model runRelativeOrientation(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                     ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                                     const ankerl::unordered_dense::set<size_t> &edges_to_optimize)
{
    (void)cam_models;
    PerformanceMeasure p("Relax runner relative");

    for (auto &node : nodes)
    {
        if (node.orientation.coeffs().hasNaN())
        {
            node.orientation = DOWN_ORIENTED_NORTH;

            // add just one image at a time with a bad initial orientation, then force an
            // optimize otherwise we could disturb the other images and end up in a bad local
            // minima the setup will already discard images which are uninitialized
            RelaxProblem rp;
            rp.setupDecompositionProblem(graph, nodes, edges_to_optimize);
            rp.solve();
        }
    }

    // finally do an optimize for the whole batch
    RelaxProblem rp;
    rp.setupDecompositionProblem(graph, nodes, edges_to_optimize);
    rp.solve();

    return rp.getSurfaceModel();
}

bool isPlaced(const NodePose &pose)
{
    return !pose.orientation.coeffs().hasNaN();
}

std::vector<NodePose> placedNeighboursInBatch(const MeasurementGraph &graph, size_t node_id,
                                              const ankerl::unordered_dense::set<size_t> &own_edges,
                                              const ankerl::unordered_dense::map<size_t, const NodePose *> &batch)
{
    std::vector<NodePose> neighbours;
    for (size_t edge_id : own_edges)
    {
        const auto *edge = graph.getEdge(edge_id);
        const size_t neighbour_id = edge->getSource() == node_id ? edge->getDest() : edge->getSource();
        auto neighbour = batch.find(neighbour_id);
        if (neighbour != batch.end() && isPlaced(*neighbour->second))
            neighbours.push_back(*neighbour->second);
    }
    return neighbours;
}

std::optional<Eigen::Quaterniond> decomposedOrientation(
    const MeasurementGraph &graph, const NodePose &node, const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
    const ankerl::unordered_dense::map<size_t, const NodePose *> &batch)
{
    constexpr double MAX_BASELINE_ANGLE = 15 * M_PI / 180, AGREEMENT_ANGLE = 3 * M_PI / 180;

    std::vector<std::pair<Eigen::Quaterniond, double>> candidates;
    for (size_t edge_id : graph.getNode(node.node_id)->getEdges())
    {
        if (!edges_to_optimize.contains(edge_id))
            continue;
        const auto *edge = graph.getEdge(edge_id);
        const bool node_is_source = edge->getSource() == node.node_id;
        auto neighbour = batch.find(node_is_source ? edge->getDest() : edge->getSource());
        if (neighbour == batch.end() || !isPlaced(*neighbour->second))
            continue;
        const NodePose &other = *neighbour->second;

        const Eigen::Vector3d baseline =
            (node_is_source ? other.position - node.position : node.position - other.position);
        double best_angle = MAX_BASELINE_ANGLE;
        std::optional<Eigen::Quaterniond> best;
        for (const auto &pose : edge->payload.relative_poses)
        {
            if (pose.score <= 0 || !pose.orientation.coeffs().allFinite() || !pose.position.allFinite())
                continue;
            const Eigen::Quaterniond relative = pose.orientation.normalized();
            const Eigen::Quaterniond source = node_is_source ? other.orientation * relative : other.orientation;
            const double cosine = std::abs((source.inverse() * baseline).normalized().dot(pose.position.normalized()));
            const double angle = std::acos(std::min(cosine, 1.0));
            if (angle < best_angle)
            {
                best_angle = angle;
                best = node_is_source ? source : other.orientation * relative.inverse();
            }
        }
        if (best)
            candidates.emplace_back(*best, edge->payload.inlier_matches.size());
    }

    double best_weight = 0;
    std::optional<Eigen::Quaterniond> mean;
    for (const auto &[pivot, _] : candidates)
    {
        double weight = 0;
        Eigen::Vector4d sum = Eigen::Vector4d::Zero();
        for (const auto &[q, w] : candidates)
            if (pivot.angularDistance(q) < AGREEMENT_ANGLE)
            {
                weight += w;
                sum += (pivot.dot(q) < 0 ? -w : w) * q.coeffs();
            }
        if (weight > best_weight)
        {
            best_weight = weight;
            mean = Eigen::Quaterniond(sum.normalized());
        }
    }
    return mean;
}

void solveAloneAgainstPlacedNeighbours(const MeasurementGraph &graph, NodePose &node,
                                       ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                                       const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
                                       const RelaxOptionSet &options,
                                       const ankerl::unordered_dense::map<size_t, const NodePose *> &batch)
{
    std::vector<NodePose> justThis{node};
    ankerl::unordered_dense::set<size_t> own_edges;
    for (size_t edge_id : graph.getNode(node.node_id)->getEdges())
        if (edges_to_optimize.contains(edge_id))
            own_edges.insert(edge_id);
    const auto neighbours = placedNeighboursInBatch(graph, node.node_id, own_edges, batch);

    RelaxProblem rp;
    rp.setupGroundPlaneProblem(graph, justThis, cam_models, own_edges, options, neighbours);
    constexpr double LOOSE_TOLERANCE_FOR_SEEDING = 1e-3;
    rp.setFunctionTolerance(LOOSE_TOLERANCE_FOR_SEEDING);
    rp.relaxObservedModelOnly();
    rp.solve();
    node = justThis[0];
}

void solveAllPlacedTogether(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                            ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                            const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
                            const RelaxOptionSet &options)
{
    RelaxProblem rp;
    rp.setupGroundPlaneProblem(graph, nodes, cam_models, edges_to_optimize, options);
    rp.relaxObservedModelOnly();
    rp.solve();
}

void initializeOrientationsOnGroundPlane(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                         ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                                         const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
                                         const RelaxOptionSet &options)
{
    constexpr size_t min_placed_to_solve_alone = 10;

    ankerl::unordered_dense::map<size_t, const NodePose *> batch;
    for (const auto &node : nodes)
        batch.emplace(node.node_id, &node);
    size_t placed_count = std::count_if(nodes.begin(), nodes.end(), isPlaced);

    Eigen::Quaterniond previous_node_orientation = DOWN_ORIENTED_NORTH;
    for (auto &node : nodes)
    {
        if (!isPlaced(node))
        {
            auto seed = decomposedOrientation(graph, node, edges_to_optimize, batch);
            if (seed && withinNadirCone(*seed))
            {
                node.orientation = *seed;
                spdlog::debug("Seed node {}: decomposed, tilt {:.1f}°", node.node_id, tiltDegrees(node.orientation));
            }
            else
            {
                const bool from_previous = withinNadirCone(previous_node_orientation);
                node.orientation = from_previous ? previous_node_orientation : DOWN_ORIENTED_NORTH;
                const bool enough_placed_to_solve_alone =
                    graph.size_nodes() > 2 * nodes.size() || placed_count >= min_placed_to_solve_alone;
                if (enough_placed_to_solve_alone)
                    solveAloneAgainstPlacedNeighbours(graph, node, cam_models, edges_to_optimize, options, batch);
                else
                    solveAllPlacedTogether(graph, nodes, cam_models, edges_to_optimize, options);
                spdlog::debug("Seed node {}: {}, started from {}, solved {}, tilt {:.1f}°", node.node_id,
                              seed ? fmt::format("decomposed tilted {:.1f}° rejected", tiltDegrees(*seed))
                                   : std::string("no decomposition"),
                              from_previous ? "previous" : "down", enough_placed_to_solve_alone ? "alone" : "together",
                              tiltDegrees(node.orientation));
            }
            placed_count++;
        }
        previous_node_orientation = node.orientation;
    }
}

surface_model runGroundPlane(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                             ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                             const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
                             const RelaxOptionSet &options)
{
    PerformanceMeasure p("Relax runner ground plane");
    initializeOrientationsOnGroundPlane(graph, nodes, cam_models, edges_to_optimize, options);

    RelaxProblem rp;
    rp.setupGroundPlaneProblem(graph, nodes, cam_models, edges_to_optimize, options);
    rp.relaxObservedModelOnly();
    rp.solve();

    return rp.getSurfaceModel();
}

surface_model runGroundMesh(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                            ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                            const ankerl::unordered_dense::set<size_t> &edges_to_optimize, const RelaxConfig &config,
                            const std::vector<surface_model> &previousSurfaces)
{
    PerformanceMeasure p("Relax runner ground mesh setup");
    RelaxProblem rp;
    rp.setupGroundMeshProblem(graph, nodes, cam_models, edges_to_optimize, config.options, previousSurfaces,
                              config.ground_mesh_grid_fraction);
    p.reset("Relax runner ground mesh solve");
    rp.relaxObservedModelOnly();
    rp.solve();

    p.reset("Relax runner ground mesh result");
    return rp.getSurfaceModel();
}

surface_model runPoints(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                        ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                        const ankerl::unordered_dense::set<size_t> &edges_to_optimize, const RelaxOptionSet &options)
{
    PerformanceMeasure p("Relax runner 3d points");
    RelaxProblem rp;
    rp.setup3dPointProblem(graph, nodes, cam_models, edges_to_optimize, options);
    rp.relaxObservedModelOnly();
    rp.solve();

    return rp.getSurfaceModel();
}

surface_model runTriangulatedRays(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                                  ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                                  const ankerl::unordered_dense::set<size_t> &edges_to_optimize,
                                  const RelaxOptionSet &options)
{
    PerformanceMeasure p("Relax runner triangulated rays init");
    RelaxOptionSet orientation_options = options;
    orientation_options.set(Option::POSITION, false);

    initializeOrientationsOnGroundPlane(graph, nodes, cam_models, edges_to_optimize, orientation_options);

    p.reset("Relax runner triangulated rays solve");
    RelaxProblem rp;
    rp.setupTriangulatedRaysProblem(graph, nodes, cam_models, edges_to_optimize, orientation_options);
    rp.solve();

    if (!options.hasAll({Option::POSITION}))
        return rp.getSurfaceModel();

    RelaxProblem position_rp;
    position_rp.setupTriangulatedRaysProblem(graph, nodes, cam_models, edges_to_optimize, options);
    position_rp.solve();
    return position_rp.getSurfaceModel();
}

} // namespace

namespace opencalibration
{

surface_model relax(const MeasurementGraph &graph, std::vector<NodePose> &nodes,
                    ankerl::unordered_dense::map<size_t, CameraModel> &cam_models,
                    const ankerl::unordered_dense::set<size_t> &edges_to_optimize, const RelaxConfig &config,
                    const std::vector<surface_model> &previousSurfaces)
{
    if (config.options.get(Option::GROUND_MESH))
        return runGroundMesh(graph, nodes, cam_models, edges_to_optimize, config, previousSurfaces);
    if (config.options.get(Option::POINTS_3D))
        return runPoints(graph, nodes, cam_models, edges_to_optimize, config.options);
    if (config.options.get(Option::TRIANGULATED_RAYS))
        return runTriangulatedRays(graph, nodes, cam_models, edges_to_optimize, config.options);
    if (config.options.get(Option::GROUND_PLANE))
        return runGroundPlane(graph, nodes, cam_models, edges_to_optimize, config.options);
    return runRelativeOrientation(graph, nodes, cam_models, edges_to_optimize);
}

} // namespace opencalibration
