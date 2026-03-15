#pragma once

#include <eigen3/Eigen/Core>

namespace opencalibration
{

struct correspondence
{
    Eigen::Vector3d measurement1;
    Eigen::Vector3d measurement2;
    double quality{0}; // lower = better match (descriptor distance), 0 = not set
};

} // namespace opencalibration
