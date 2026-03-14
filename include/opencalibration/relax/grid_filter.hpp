#pragma once

#include <ankerl/unordered_dense.h>
#include <math.h>
#include <tuple>
#include <vector>

namespace opencalibration
{

inline uint64_t gridCellKey(int i, int j)
{
    return (static_cast<uint64_t>(i) << 32) | static_cast<uint32_t>(j);
}

template <typename T> class GridFilter
{

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
        uint64_t index = gridCellKey(std::floor(x / _grid_resolution), std::floor(y / _grid_resolution));
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
