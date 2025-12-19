#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <vector>

namespace opencalibration
{

template <typename T, size_t D> class KMeans
{
    using point_vec = std::vector<std::pair<std::array<double, D>, T>>;

  public:
    struct cluster
    {
        std::array<double, D> centroid{};
        std::vector<std::pair<std::array<double, D>, T>> points;

        bool operator<(const cluster &c)
        {
            return points.size() < c.points.size();
        }
    };

    KMeans(size_t num_clusters)
    {
        _clusters.resize(num_clusters);
    }

    void reset(size_t num_clusters)
    {
        _clusters.clear();
        _clusters.resize(num_clusters);
        _initialized = false;
    }

    void add(const std::array<double, D> &location, const T &value)
    {

        if (_initialized)
        {
            size_t nearest_cluster_index = nearest_cluster(location);
            add_shift_centroid(_clusters[nearest_cluster_index].centroid, location,
                               _clusters[nearest_cluster_index].points.size());
            _clusters[nearest_cluster_index].points.emplace_back(location, value);
        }
        else
        {
            _clusters[0].points.emplace_back(location, value);
        }
    }

    bool iterate()
    {
        if (!_initialized)
        {
            if (initialize())
                return true; // first iteration just initializes on furthest points
            else
                return false;
        }

        reassign_centroids();

        reassign_points();

        // calculate new centroids
        recalculate_centroids();

        std::sort(_clusters.begin(), _clusters.end());
        return true;
    }

    const std::vector<cluster> &getClusters()
    {
        return _clusters;
    }

  private:
    bool initialize()
    {
        if (_clusters[0].points.size() < _clusters.size())
        {
            return false;
        }

        point_vec points = collect_all_points();

        // K-center initialization
        std::vector<size_t> seeds;
        seeds.reserve(_clusters.size());
        seeds.push_back(0);

        std::vector<double> min_dists(points.size(), std::numeric_limits<double>::max());

        for (size_t k = 1; k < _clusters.size(); ++k)
        {
            const auto &last_seed_loc = points[seeds.back()].first;
            size_t furthest_idx = 0;
            double max_min_dist = -1.0;

            for (size_t i = 0; i < points.size(); ++i)
            {
                min_dists[i] = std::min(min_dists[i], distance_sq(points[i].first, last_seed_loc));

                if (min_dists[i] > max_min_dist)
                {
                    max_min_dist = min_dists[i];
                    furthest_idx = i;
                }
            }
            seeds.push_back(furthest_idx);
        }

        for (size_t k = 0; k < _clusters.size(); ++k)
        {
            _clusters[k].centroid = points[seeds[k]].first;
        }

        for (const auto &p : points)
        {
            size_t best_idx = 0;
            double min_dist = std::numeric_limits<double>::infinity();
            for (size_t k = 0; k < _clusters.size(); ++k)
            {
                double d = distance_sq(p.first, _clusters[k].centroid);
                if (d < min_dist)
                {
                    min_dist = d;
                    best_idx = k;
                }
            }
            _clusters[best_idx].points.push_back(p);
        }

        recalculate_centroids();
        std::sort(_clusters.begin(), _clusters.end());

        _initialized = true;
        return true;
    }

    std::vector<std::pair<std::array<double, D>, T>> collect_all_points()
    {
        std::vector<std::pair<std::array<double, D>, T>> points;
        size_t num_points = 0;
        for (const auto &c : _clusters)
        {
            num_points += c.points.size();
        }
        points.reserve(num_points);
        for (auto &c : _clusters)
        {
            points.insert(points.end(), c.points.begin(), c.points.end());
            c.points.clear();
        }
        return points;
    }

    void recalculate_centroids()
    {
        for (auto &c : _clusters)
        {
            if (c.points.empty())
                continue;
            std::fill(c.centroid.begin(), c.centroid.end(), 0);
            for (const auto &p : c.points)
            {
                for (size_t i = 0; i < D; i++)
                {
                    c.centroid[i] += p.first[i];
                }
            }
            for (size_t i = 0; i < D; i++)
            {
                c.centroid[i] /= c.points.size();
            }
        }
    }

    size_t nearest_cluster(const std::array<double, D> &location)
    {
        size_t nearest_cluster_index = 0;
        double nearest_distance = std::numeric_limits<double>::infinity();

        for (size_t i = 0; i < _clusters.size(); i++)
        {
            const double distance = distance_sq(location, _clusters[i].centroid);
            if (distance < nearest_distance)
            {
                nearest_cluster_index = i;
                nearest_distance = distance;
            }
            if (!_initialized && _clusters[i].points.size() == 0)
            {
                nearest_cluster_index = i;
                nearest_distance = 0;
            }
        }
        return nearest_cluster_index;
    }

    void reassign_centroids()
    {
        size_t num_to_redistribute = 0;
        const double ratio = 2.71828;

        for (; num_to_redistribute < _clusters.size() / 2; num_to_redistribute++)
        {
            if (_clusters[num_to_redistribute].points.size() * ratio >
                _clusters[_clusters.size() - 1 - num_to_redistribute].points.size())
            {
                break;
            }
        }

        // redistribute cluster centers from small clusters to big clusters
        for (size_t i = 0; i < num_to_redistribute; i++)
        {
            for (size_t j = 0; j < D; j++)
            {
                const double sign = (i + j) % 2 == 0 ? 1 : -1;
                _clusters[i].centroid[j] = _clusters[_clusters.size() - 1 - i].centroid[j] * (1 + sign * 1e-9);
            }
        }
    }

    void reassign_points()
    {
        // take out all points, leaving centroids where they are
        point_vec points = collect_all_points();

        // re-assign points to closest centroid
        for (const auto &p : points)
        {
            size_t nearest_cluster_index = nearest_cluster(p.first);
            _clusters[nearest_cluster_index].points.push_back(p);
        }
    }

    void add_shift_centroid(std::array<double, D> &centroid, const std::array<double, D> &new_value,
                            size_t centroid_points)
    {
        for (size_t i = 0; i < D; i++)
        {
            centroid[i] = (new_value[i] * 1. + centroid[i] * centroid_points) / (centroid_points + 1);
        }
    }

    double distance_sq(const std::array<double, D> &a, const std::array<double, D> &b)
    {
        auto sqr = [](double v) { return v * v; };
        double dist = 0;
        for (size_t i = 0; i < D; i++)
        {
            dist += sqr(a[i] - b[i]);
        }
        return dist;
    }

    std::default_random_engine _random_generator{42};
    std::vector<cluster> _clusters;
    bool _initialized{false};
};

} // namespace opencalibration
