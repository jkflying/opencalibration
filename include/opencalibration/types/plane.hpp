#pragma once

#include <eigen3/Eigen/Core>

namespace opencalibration
{

template <typename T = double> struct plane_norm_offset
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix<T, 3, 1> norm{T(NAN), T(NAN), T(NAN)};
    Eigen::Matrix<T, 3, 1> offset{T(NAN), T(NAN), T(NAN)};
};

template <typename T = double> struct plane_3_corners
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix<T, 3, 1> corner[3] = {{T(NAN), T(NAN), T(NAN)}, {T(NAN), T(NAN), T(NAN)}, {T(NAN), T(NAN), T(NAN)}};
};
} // namespace opencalibration
