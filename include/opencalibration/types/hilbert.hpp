#pragma once

#include <eigen3/Eigen/Geometry>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace opencalibration
{

inline uint32_t xy2d(int order, int x, int y)
{
    uint32_t d = 0;
    for (int s = order / 2; s > 0; s /= 2)
    {
        int rx = (x & s) > 0 ? 1 : 0;
        int ry = (y & s) > 0 ? 1 : 0;
        d += s * s * ((3 * rx) ^ ry);
        if (ry == 0)
        {
            if (rx == 1)
            {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::swap(x, y);
        }
    }
    return d;
}

inline std::vector<size_t> hilbertOrder(const std::vector<Eigen::Vector2d> &points, const Eigen::AlignedBox2d &bounds)
{
    constexpr int N = 1 << 15;
    const Eigen::Vector2d scale = (N - 1) / bounds.sizes().cwiseMax(1e-9).array();
    std::vector<std::pair<uint32_t, size_t>> keyed(points.size());
    for (size_t i = 0; i < points.size(); i++)
    {
        const Eigen::Vector2d g = (points[i] - bounds.min()).cwiseProduct(scale);
        keyed[i] = {
            xy2d(N, std::clamp(static_cast<int>(g.x()), 0, N - 1), std::clamp(static_cast<int>(g.y()), 0, N - 1)), i};
    }
    std::sort(keyed.begin(), keyed.end());
    std::vector<size_t> order(points.size());
    for (size_t i = 0; i < keyed.size(); i++)
        order[i] = keyed[i].second;
    return order;
}

} // namespace opencalibration
