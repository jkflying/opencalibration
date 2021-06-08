#pragma once

#include <eigen3/Eigen/Core>

namespace opencalibration
{
template <typename T> struct ray
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix<T, 3, 1> dir{T(NAN), T(NAN), T(NAN)};
    Eigen::Matrix<T, 3, 1> offset{T(NAN), T(NAN), T(NAN)};
};

using ray_d = ray<double>;

} // namespace opencalibration
