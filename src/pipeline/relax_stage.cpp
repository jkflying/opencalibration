#include <opencalibration/pipeline/relax_stage.hpp>

#include <opencalibration/geometry/spectral_cluster.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/relax/relax_group.hpp>

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
                      const jk::tree::KDTree<size_t, 3> &imageGPSLocations, bool final_global_relax)
{
    PerformanceMeasure p("Relax init");
    _groups.clear();

    std::vector<size_t> actual_node_ids = node_ids;

    if (final_global_relax)
    {
        actual_node_ids.clear();
        actual_node_ids.reserve(graph.size_nodes());
        for (auto iter = graph.nodebegin(); iter != graph.nodeend(); ++iter)
        {
            actual_node_ids.push_back(iter->first);
        }
    }

    const size_t num_groups =
        final_global_relax ? std::max<size_t>(1, static_cast<size_t>(std::ceil(actual_node_ids.size() / 50))) : 1;

    _k_groups->reset(num_groups);

    for (size_t node_id : actual_node_ids)
    {
        auto location = to_array(graph.getNode(node_id)->payload.position);
        _k_groups->add(location, node_id);
    }

    if (num_groups > 1)
    {
        spdlog::info("Splitting relax into {} group(s)", num_groups);

        for (size_t i = 0; i < actual_node_ids.size(); i++)
        {
            size_t node_id = actual_node_ids[i];
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

    for (const auto &group : _k_groups->getClusters())
    {
        std::vector<size_t> group_ids;
        group_ids.reserve(group.points.size());
        for (const auto &p : group.points)
        {
            group_ids.push_back(p.second);
        }
        spdlog::info("Group added with {} nodes", group_ids.size());
        _groups.emplace_back();
        RelaxOptionSet options({Option::ORIENTATION});
        if (final_global_relax)
        {
            options.set(Option::POINTS_3D, true);
        }
        _groups.back().init(graph, group_ids, imageGPSLocations, graph_connection_depth, options);
    }
    // k-means were smallest to biggest, but we want to process the big ones first to improve load balancing on really
    // large problem
    std::reverse(_groups.begin(), _groups.end());
}

std::vector<std::function<void()>> RelaxStage::get_runners(const MeasurementGraph &graph)
{
    std::vector<std::function<void()>> funcs;
    for (auto &g : _groups)
    {
        funcs.push_back([&]() { g.run(graph); });
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
    return optimized_ids;
}

} // namespace opencalibration
