#pragma once

#include <opencalibration/geometry/KMeans.hpp>

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <eigen3/Eigen/SparseCore>

#include <array>
#include <queue>
#include <unordered_set>
#include <vector>

namespace opencalibration
{

template <typename T, size_t D> class SpectralClustering
{
    template <size_t N> std::array<double, N> toArray(Eigen::Matrix<double, N, 1> e)
    {
        std::array<double, N> res;
        std::copy_n(e.data(), N, res.begin());
        return res;
    }

  public:
    using KMeans = opencalibration::KMeans<T, D>;
    SpectralClustering(size_t num_clusters) : _kmeans(num_clusters)
    {
    }

    void reset(size_t num_clusters)
    {
        _kmeans.reset(num_clusters);
        _items.clear();
        _links.clear();
        _sub_clusters.clear();
        _combined_view.clear();
    }

    void add(const std::array<double, D> &location, const T &value)
    {
        _items.emplace_back(location, value);
    }

    void addLink(const T &val1, const T &val2, double weight)
    {
        _links.emplace_back(val1, val2, weight);
    }

    bool spectralize()
    {
        bool spectralized = false;
        if (_kmeans.getClusters().size() > 1)
        {

            using Sparse = Eigen::SparseMatrix<double>;
            std::unordered_map<T, size_t> reverse_item_lookup;
            Sparse degree(_items.size(), _items.size());
            degree.reserve(_items.size());
            for (size_t i = 0; i < _items.size(); i++)
            {
                reverse_item_lookup.emplace(_items[i].second, i);
                degree.insert(i, i) = 0;
            }
            degree.makeCompressed();

            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(_links.size());
            for (const auto &link : _links)
            {
                auto index_it_0 = reverse_item_lookup.find(std::get<0>(link));
                auto index_it_1 = reverse_item_lookup.find(std::get<1>(link));
                if (index_it_0 != reverse_item_lookup.end() && index_it_1 != reverse_item_lookup.end())
                {
                    size_t i0 = index_it_0->second;
                    size_t i1 = index_it_1->second;
                    double w = std::get<2>(link);
                    triplets.emplace_back(i0, i1, w);
                    triplets.emplace_back(i1, i0, w);
                    degree.coeffRef(i0, i0) += w;
                    degree.coeffRef(i1, i1) += w;
                }
            }
            if ((degree.diagonal().array() == 0.).count() > 0)
            {
                // graph has isolated nodes
                return false;
            }

            Sparse adjacency(degree.rows(), degree.cols());
            adjacency.setFromTriplets(triplets.begin(), triplets.end());

            // Split into subclusters based on connectivity
            std::vector<std::vector<size_t>> components;
            {
                std::vector<bool> visited(_items.size(), false);
                for (size_t start = 0; start < _items.size(); start++)
                {
                    if (visited[start])
                        continue;
                    components.emplace_back();
                    std::queue<size_t> queue;
                    queue.push(start);
                    visited[start] = true;
                    while (!queue.empty())
                    {
                        size_t node = queue.front();
                        queue.pop();
                        components.back().push_back(node);
                        for (Sparse::InnerIterator it(adjacency, node); it; ++it)
                        {
                            if (!visited[it.index()])
                            {
                                visited[it.index()] = true;
                                queue.push(it.index());
                            }
                        }
                    }
                }
            }

            if (components.size() > 1)
                return spectralizeComponents(components);

            const Sparse laplacian = degree - adjacency;

            // Use the Ng, Jordan and Weiss (2002) formulation in order to use the symmetric eigen solver
            // which converges faster and doesn't have imaginary components in the eigenvectors
            Eigen::VectorXd inverse_sqrt_degree = degree.diagonal().array().sqrt().inverse();

            const Sparse normalizedAdjacency =
                inverse_sqrt_degree.asDiagonal() * adjacency * inverse_sqrt_degree.asDiagonal();
            Sparse identity = degree;
            identity.setIdentity();
            const Sparse normalizedLaplacian = identity - normalizedAdjacency;

            using Op = Spectra::SparseSymMatProd<double>;
            Op op(normalizedLaplacian);
            Spectra::SymEigsSolver<Op> eigen_solver(op, D + 1, D + 3);

            eigen_solver.init();
            int nconv = eigen_solver.compute(Spectra::SortRule::SmallestMagn);

            if (eigen_solver.info() == Spectra::CompInfo::Successful && nconv == D + 1)
            {

                const Eigen::MatrixXd evectors = eigen_solver.eigenvectors();

                for (size_t i = 0; i < _items.size(); i++)
                {
                    auto location = toArray<D>(evectors.block<1, D>(i, 0).normalized().transpose());
                    T identifier = _items[i].second;
                    _kmeans.add(location, identifier);
                }
                spectralized = true;
            }
        }
        return spectralized;
    }

    void fallback()
    {
        for (const auto &item : _items)
        {
            _kmeans.add(item.first, item.second);
        }
    }

    void iterate()
    {
        if (!_sub_clusters.empty())
        {
            for (auto &sub : _sub_clusters)
                sub.iterate();
            rebuildCombinedView();
        }
        else
        {
            _kmeans.iterate();
        }
    }

    const std::vector<typename KMeans::cluster> &getClusters() const
    {
        if (!_sub_clusters.empty())
            return _combined_view;
        return _kmeans.getClusters();
    }

  private:
    // Allocate num_clusters across components, min 1 each, extras to the largest.
    bool spectralizeComponents(const std::vector<std::vector<size_t>> &components)
    {
        const size_t num_clusters = _kmeans.getClusters().size();

        std::vector<size_t> alloc(components.size(), 1);
        for (size_t extra = components.size(); extra < num_clusters; extra++)
        {
            size_t best = 0;
            double best_ratio = 0;
            for (size_t i = 0; i < components.size(); i++)
            {
                double ratio = static_cast<double>(components[i].size()) / alloc[i];
                if (ratio > best_ratio)
                {
                    best_ratio = ratio;
                    best = i;
                }
            }
            alloc[best]++;
        }

        for (size_t c = 0; c < components.size(); c++)
        {
            std::unordered_set<T> comp_ids;
            for (size_t idx : components[c])
                comp_ids.insert(_items[idx].second);

            _sub_clusters.emplace_back(alloc[c]);
            auto &sub = _sub_clusters.back();

            for (size_t idx : components[c])
                sub.add(_items[idx].first, _items[idx].second);
            for (const auto &link : _links)
                if (comp_ids.count(std::get<0>(link)) && comp_ids.count(std::get<1>(link)))
                    sub.addLink(std::get<0>(link), std::get<1>(link), std::get<2>(link));

            bool sub_spec = (alloc[c] > 1) && sub.spectralize();
            if (!sub_spec)
                sub.fallback();
        }

        rebuildCombinedView();
        return true;
    }

    void rebuildCombinedView()
    {
        _combined_view.clear();
        for (const auto &sub : _sub_clusters)
            for (const auto &cluster : sub.getClusters())
                _combined_view.push_back(cluster);
        std::sort(_combined_view.begin(), _combined_view.end());
    }

    std::vector<std::pair<std::array<double, D>, T>> _items;
    std::vector<std::tuple<T, T, double>> _links;
    KMeans _kmeans;
    std::vector<SpectralClustering<T, D>> _sub_clusters;
    std::vector<typename KMeans::cluster> _combined_view;
};

} // namespace opencalibration
