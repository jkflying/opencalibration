#include <opencalibration/pipeline/relax_stage.hpp>

#include <jk/KMeans.h>
#include <opencalibration/performance/performance.hpp>

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <eigen3/Eigen/SparseCore>

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

    const size_t num_groups =
        final_global_relax ? std::max<size_t>(1, static_cast<size_t>(std::ceil(actual_node_ids.size() / 50))) : 1;

    jk::tree::KMeans<size_t, 3> k_groups(num_groups);

    if (num_groups > 1)
    {
        spdlog::info("Splitting relax into {} group(s)", num_groups);

        // spectral cluster based on mincut of edges rather than clustering on GPS locations
        // TODO: weight edges that were cut in previous iterations higher so that the clusters change each iteration
        std::vector<Eigen::Triplet<double>> triplets;
        if (graph.size_nodes() > 0)
        {
            triplets.reserve(graph.size_edges() * actual_node_ids.size() * 20 / graph.size_nodes());
        }

        std::unordered_map<size_t, size_t> reverse_node_id_lookup;
        Eigen::SparseMatrix<double> degree(actual_node_ids.size(), actual_node_ids.size());
        degree.reserve(actual_node_ids.size());
        for (size_t i = 0; i < actual_node_ids.size(); i++)
        {
            reverse_node_id_lookup.emplace(actual_node_ids[i], i);
            degree.insert(i, i) = 0;
        }
        degree.makeCompressed();

        for (size_t i = 0; i < actual_node_ids.size(); i++)
        {
            size_t node_id = actual_node_ids[i];
            for (size_t edge_id : graph.getNode(node_id)->getEdges())
            {
                const auto *edge = graph.getEdge(edge_id);
                auto source_index_it = reverse_node_id_lookup.find(edge->getSource());
                auto dest_index_it = reverse_node_id_lookup.find(edge->getDest());
                if (source_index_it != reverse_node_id_lookup.end() && dest_index_it != reverse_node_id_lookup.end())
                {
                    size_t si = source_index_it->second;
                    size_t di = dest_index_it->second;
                    triplets.emplace_back(si, di, 1);
                    triplets.emplace_back(di, si, 1);
                    degree.coeffRef(si, si) += 1;
                    degree.coeffRef(di, di) += 1;
                }
            }
        }
        Eigen::SparseMatrix<double> adjacency(actual_node_ids.size(), actual_node_ids.size());
        adjacency.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::SparseMatrix<double> laplacian = degree - adjacency;

        using Op = Spectra::SparseSymMatProd<double>;
        const size_t dimensions = 3;
        Op op(laplacian);
        Spectra::SymEigsSolver<double, Spectra::LARGEST_MAGN, Op> eigen_solver(&op, dimensions, dimensions + 1);

        eigen_solver.init();
        int nconv = eigen_solver.compute();

        // Retrieve results
        Eigen::VectorXd evalues;
        Eigen::MatrixXd evectors;
        if (eigen_solver.info() == Spectra::SUCCESSFUL)
        {
            evalues = eigen_solver.eigenvalues();
            evectors = eigen_solver.eigenvectors();
            spdlog::info("{} eigenvalues found, for eigenvectors of length {}", nconv, evectors.rows());

            // std::cout << "Laplacian:\n" << laplacian.toDense() << std::endl;
            // std::cout << nconv << " eigenvalues found:\n" << evalues.transpose() << std::endl;
            // std::cout << "Eigenvectors:\n" << evectors << std::endl;

            for (size_t i = 0; i < actual_node_ids.size(); i++)
            {
                auto location = to_array(evectors.row(i));
                size_t node_id = actual_node_ids[i];
                k_groups.add(location, node_id);
            }
        }

        else
        {
            for (size_t node_id : actual_node_ids)
            {
                auto location = to_array(graph.getNode(node_id)->payload.position);
                k_groups.add(location, node_id);
            }
        }

        if (num_groups > 1)
        {
            for (int i = 0; i < 10; i++) // 10 iterations should be enough for anybody
            {
                k_groups.iterate();
            }
        }
    }
    else
    {

        for (size_t node_id : actual_node_ids)
        {
            auto location = to_array(graph.getNode(node_id)->payload.position);
            k_groups.add(location, node_id);
        }
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
