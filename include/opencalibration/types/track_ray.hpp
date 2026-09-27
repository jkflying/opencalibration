#pragma once

#include <eigen3/Eigen/Geometry>

namespace opencalibration
{

struct TrackRay
{
    size_t node_id;
    size_t feature_index;
    size_t camera_model_id;
    Eigen::Vector3d camera_loc;
    Eigen::Vector3d camera_ray;
    Eigen::Vector2d pixel;
    Eigen::Quaterniond orientation;
    double *pose_ptr;
    bool optimize;
    double inverse_sigma;
};
} // namespace opencalibration
