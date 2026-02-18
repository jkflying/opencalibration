#pragma once

#include <ankerl/unordered_dense.h>
#include <math.h>
#include <tuple>
#include <vector>

namespace opencalibration
{

template <typename T> class GridFilter
{

    inline uint64_t key(int i, int j)
    {
        return ((uint64_t)i << 32) | (unsigned int)j;
    }

  public:
    GridFilter(double grid_resolution = 0.075) : _grid_resolution(grid_resolution)
    {
        _map.reserve(static_cast<size_t>(1 / (grid_resolution * grid_resolution)));
    }

    void setResolution(double grid_resolution)
    {
        if (_map.size() == 0)
        {
            _grid_resolution = grid_resolution;
        }
    }

    void addMeasurement(double x, double y, double score, const T &value)
    {
        uint64_t index = key(std::floor(x / _grid_resolution), std::floor(y / _grid_resolution));
        auto iter = _map.find(index);
        if (iter == _map.end())
        {
            _map.emplace(index, std::make_pair(score, value));
            _best.insert(value);
        }
        else
        {
            if (iter->second.first < score)
            {
                _best.erase(iter->second.second);
                iter->second = std::make_pair(score, value);
                _best.insert(value);
            }
        }
    }

    [[nodiscard]] const ankerl::unordered_dense::set<T> &getBestMeasurementsPerCell() const
    {
        return _best;
    }

  private:
    double _grid_resolution;
    ankerl::unordered_dense::map<uint64_t, std::pair<double, T>> _map;
    ankerl::unordered_dense::set<T> _best;
};

} // namespace opencalibration
