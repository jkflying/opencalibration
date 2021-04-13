#pragma once

#include <eigen3/Eigen/Core>

namespace opencalibration
{
using point_cloud = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
}
