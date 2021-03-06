#pragma once

#include <eigen3/Eigen/Geometry>

namespace opencalibration
{

struct NodePose
{
    size_t node_id;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d position;
};
} // namespace opencalibration
