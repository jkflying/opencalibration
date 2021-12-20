#pragma once

#include <array>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace opencalibration
{

inline bool anticlockwise(const std::array<Eigen::Vector3d, 3> &points)
{
    double crossZ = (points[1] - points[0]).cross(points[2] - points[0]).z();
    return crossZ < 0;
}

} // namespace opencalibration
