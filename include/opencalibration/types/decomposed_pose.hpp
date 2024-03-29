#pragma once

#include <eigen3/Eigen/Geometry>

namespace opencalibration
{
struct decomposed_pose
{
    Eigen::Quaterniond orientation{NAN, NAN, NAN, NAN};
    Eigen::Vector3d position{NAN, NAN, NAN};
    int score{0};

    bool operator==(const decomposed_pose &other) const
    {
        return score == other.score &&
               ((orientation.coeffs().array().isNaN().all() && other.orientation.coeffs().array().isNaN().all()) ||
                orientation.coeffs() == other.orientation.coeffs()) &&
               ((position.array().isNaN().all() && other.position.array().isNaN().all()) || position == other.position);
    }
};

} // namespace opencalibration
