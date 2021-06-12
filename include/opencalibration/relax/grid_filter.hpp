#pragma once

#include <math.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
// #include >

namespace opencalibration
{

template <typename T> class GridFilter
{

    inline size_t key(int i, int j)
    {
        return (size_t)i << 32 | (unsigned int)j;
    }

  public:
    GridFilter(double grid_resolution = 0.1) : _grid_resolution(grid_resolution)
    {
    }

    void addMeasurement(double x, double y, double score, const T &value)
    {
        size_t index = key(std::floor(x / _grid_resolution), std::floor(y / _grid_resolution));
        _map[index].push_back(std::make_pair(score, value));
    }

    std::unordered_set<T> getBestMeasurementsPerCell() const
    {
        std::unordered_set<T> best_measurements;
        best_measurements.reserve(_map.size()); // 1 per bucket

        for (const auto &index_data : _map)
        {
            double best_score = -std::numeric_limits<double>::infinity();
            const T *best_value = nullptr;

            for (const auto &score_value : index_data.second)
            {
                if (score_value.first > best_score)
                {
                    best_score = score_value.first;
                    best_value = &score_value.second;
                }
            }

            if (best_value != nullptr)
            {
                best_measurements.insert(*best_value);
            }
        }
        return best_measurements;
    }

  private:
    double _grid_resolution;
    std::unordered_map<size_t, std::vector<std::pair<double, T>>> _map;
};

} // namespace opencalibration
