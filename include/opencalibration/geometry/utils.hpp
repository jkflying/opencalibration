#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace opencalibration
{

inline bool anticlockwise(const std::array<Eigen::Vector3d, 3> &points)
{
    double crossZ = (points[1] - points[0]).cross(points[2] - points[0]).z();
    return crossZ < 0;
}

inline double median(std::vector<double> values)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

} // namespace opencalibration
