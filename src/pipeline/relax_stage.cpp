#include <opencalibration/pipeline/relax_stage.hpp>

#include <opencalibration/geometry/spectral_cluster.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/relax/relax_group.hpp>
#include <opencalibration/surface/refine_mesh.hpp>

#include <spdlog/spdlog.h>

namespace
{
std::array<double, 3> to_array(const Eigen::Vector3d &v)
{
    return {v.x(), v.y(), v.z()};
}
} // namespace

namespace opencalibration
{

RelaxStage::RelaxStage() : _k_groups(new SpectralClustering<size_t, 3>(0))
{
}
RelaxStage::~RelaxStage()
{
}

void RelaxStage::init(const MeasurementGraph &graph, const std::vector<size_t> &node_ids,
                      const jk::tree::KDTree<size_t, 2> &imageGPSLocations, bool relax_all, bool disable_parallelism,
                      const RelaxOptionSet &options)
{
    PerformanceMeasure p("Relax init");
    spdlog::info("Initializing relax with options: {}", toString(options));
    _groups.clear();

    std::vector<size_t> actual_node_ids = node_ids;

    if (relax_all)
    {
        actual_node_ids.clear();
        actual_node_ids.reserve(graph.size_nodes());
        for (auto iter = graph.cnodebegin(); iter != graph.cnodeend(); ++iter)
        {
            actual_node_ids.push_back(iter->first);
        }
    }

    const bool global_params = options.hasAny({Option::FOCAL_LENGTH, Option::PRINCIPAL_POINT,
                                               Option::LENS_DISTORTIONS_RADIAL, Option::LENS_DISTORTIONS_TANGENTIAL});

    const int optimal_cluster_size = global_params ? 150 : 50;

    const size_t num_groups =
        disable_parallelism
            ? 1
            : std::max<size_t>(1, static_cast<size_t>(std::floor(actual_node_ids.size() / optimal_cluster_size)));

    _k_groups->reset(num_groups);

    for (size_t node_id : actual_node_ids)
    {
        auto location = to_array(graph.getNode(node_id)->payload.position);
        _k_groups->add(location, node_id);
    }

    if (num_groups > 1)
    {
        spdlog::info("Splitting relax into {} group(s)", num_groups);

        for (size_t node_id : actual_node_ids)
        {
            _k_groups->addLink(node_id, node_id, 0.1);
            for (size_t edge_id : graph.getNode(node_id)->getEdges())
            {
                const auto *edge = graph.getEdge(edge_id);
                // TODO: add weight for edges which were cut during the last graph partitioning, so they are less likely
                // to get cut next time
                _k_groups->addLink(edge->getSource(), edge->getDest(), 1);
            }
        }
        if (!_k_groups->spectralize())
        {
            spdlog::info("Spectralize failed");
            _k_groups->fallback();
        }
        for (int i = 0; i < 10; i++) // 10 iterations should be enough for anybody
        {
            _k_groups->iterate();
        }
    }
    else
    {
        _k_groups->fallback();
    }

    size_t graph_connection_depth = num_groups > 1 ? 0 : 2;
    const auto &cluster = _k_groups->getClusters();

    // k-means were smallest to biggest, but we want to process the big ones first to improve load balancing on really
    // large problem
    for (auto it = cluster.rbegin(); it != cluster.rend(); ++it)
    {
        const auto &group = *it;
        std::vector<size_t> group_ids;
        group_ids.reserve(group.points.size());
        for (const auto &p : group.points)
        {
            group_ids.push_back(p.second);
        }
        _groups.emplace_back();
        _groups.back().init(graph, group_ids, imageGPSLocations, graph_connection_depth, options);
    }
}

void RelaxStage::trim_groups(size_t max_size)
{
    while (_groups.size() > max_size)
    {
        _groups.pop_back();
    }
}

std::vector<std::function<void()>> RelaxStage::get_runners(const MeasurementGraph &graph)
{
    std::vector<std::function<void()>> funcs;

    std::swap(_surface_models, _previous_surface_models);

    _surface_models.clear();
    _surface_models.resize(_groups.size());
    for (size_t i = 0; i < _groups.size(); i++)
    {
        auto &g = _groups[i];
        auto &s = _surface_models[i];
        auto &ps = _previous_surface_models;
        funcs.push_back([&s, &g, &graph, &ps]() { s = g.run(graph, ps); });
    }
    return funcs;
}

std::vector<std::vector<size_t>> RelaxStage::finalize(MeasurementGraph &graph)
{
    PerformanceMeasure p("Relax finalize");
    std::vector<std::vector<size_t>> optimized_ids;
    for (auto &g : _groups)
    {
        optimized_ids.emplace_back(g.finalize(graph));
    }
    _groups.clear();

    // Merge surface models from all groups if there are multiple
    if (_surface_models.size() > 1)
    {
        spdlog::info("Merging {} surface models from parallel groups", _surface_models.size());
        surface_model merged = mergeSurfaceModels(_surface_models);
        _surface_models.clear();
        _surface_models.push_back(std::move(merged));
    }

    return optimized_ids;
}

const std::vector<surface_model> &RelaxStage::getSurfaceModels()
{
    return _surface_models;
}

void RelaxStage::setSurfaceModels(std::vector<surface_model> surfaces)
{
    _surface_models = std::move(surfaces);
}

} // namespace opencalibration
