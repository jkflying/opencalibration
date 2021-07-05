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

    inline uint64_t key(int i, int j)
    {
        return ((uint64_t)i << 32) | (unsigned int)j;
    }

  public:
    GridFilter(double grid_resolution = 0.1) : _grid_resolution(grid_resolution)
    {
        _map.reserve(static_cast<size_t>(1 / (grid_resolution * grid_resolution)));
    }

    void addMeasurement(double x, double y, double score, const T &value)
    {
        uint64_t index = key(std::floor(x / _grid_resolution), std::floor(y / _grid_resolution));
        auto iter = _map.find(index);
        if (iter == _map.end())
        {
            _map.emplace(index, std::make_pair(score, value));
        }
        else
        {
            if (iter->second.first < score)
            {
                iter->second = std::make_pair(score, value);
            }
        }
    }

    std::unordered_set<T> getBestMeasurementsPerCell() const
    {
        std::unordered_set<T> best_measurements;
        best_measurements.reserve(_map.size());

        for (const auto &index_data : _map)
        {
            best_measurements.insert(index_data.second.second);
        }
        return best_measurements;
    }

  private:
    double _grid_resolution;
    std::unordered_map<uint64_t, std::pair<double, T>> _map;
};

} // namespace opencalibration
