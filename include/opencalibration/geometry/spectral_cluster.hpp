#pragma once

#include <opencalibration/geometry/KMeans.hpp>

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <eigen3/Eigen/SparseCore>

#include <array>
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
                // graph is not connected
                return false;
            }

            Sparse adjacency(degree.rows(), degree.cols());
            adjacency.setFromTriplets(triplets.begin(), triplets.end());
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
                    auto location = toArray<D>(evectors.block<1, D>(i, 0).normalized());
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
        _kmeans.iterate();
    }

    const std::vector<typename KMeans::cluster> &getClusters()
    {
        return _kmeans.getClusters();
    }

  private:
    std::vector<std::pair<std::array<double, D>, T>> _items;
    std::vector<std::tuple<T, T, double>> _links;
    KMeans _kmeans;
};

} // namespace opencalibration
