#pragma once

#include <eigen3/Eigen/Core>

namespace opencalibration
{

struct correspondence
{
    Eigen::Vector3d measurement1;
    Eigen::Vector3d measurement2;
};

} // namespace opencalibration
