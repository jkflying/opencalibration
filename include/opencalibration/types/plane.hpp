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

template <typename T> struct plane_3_corners
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::array<Eigen::Matrix<T, 3, 1>, 3> corner = {
        {{T(NAN), T(NAN), T(NAN)}, {T(NAN), T(NAN), T(NAN)}, {T(NAN), T(NAN), T(NAN)}}};
};

using plane_norm_offset_d = plane_norm_offset<double>;
using plane_3_corners_d = plane_3_corners<double>;
} // namespace opencalibration
