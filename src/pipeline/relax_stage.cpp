#include <opencalibration/pipeline/relax_stage.hpp>

#include <jk/KMeans.h>
#include <opencalibration/performance/performance.hpp>

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

    const size_t num_groups = std::max<size_t>(1, static_cast<size_t>(std::ceil(actual_node_ids.size() / 200)));

    spdlog::info("Splitting relax into {} groups", num_groups);

    jk::tree::KMeans<size_t, 3> k_groups(num_groups);

    for (size_t node_id : actual_node_ids)
    {
        auto location = to_array(graph.getNode(node_id)->payload.position);
        k_groups.add(location, node_id);
    }

    if (num_groups > 1)
        for (int i = 0; i < 10; i++) // 10 iterations should be enough for anybody
        {
            k_groups.iterate();
        }

    size_t graph_connection_depth = num_groups > 1 ? 0 : 2;

    for (const auto &group : k_groups.getClusters())
    {
        std::vector<size_t> group_ids;
        group_ids.reserve(group.points.size());
        for (const auto &p : group.points)
        {
            group_ids.push_back(p.second);
        }
        spdlog::info("Group added with {} nodes", group_ids.size());
        _groups.emplace_back();
        _groups.back().init(graph, group_ids, imageGPSLocations, graph_connection_depth,
                            final_global_relax ? RelaxGroup::RelaxType::MEASUREMENT_RELAX_POINTS
                                               : RelaxGroup::RelaxType::RELATIVE_RELAX);
    }
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

std::vector<size_t> RelaxStage::finalize(MeasurementGraph &graph)
{
    PerformanceMeasure p("Relax finalize");
    std::vector<size_t> optimized_ids;
    for (auto &g : _groups)
    {
        auto group_ids = g.finalize(graph);
        optimized_ids.insert(optimized_ids.end(), group_ids.begin(), group_ids.end());
    }
    _groups.clear();
    return optimized_ids;
}

} // namespace opencalibration
