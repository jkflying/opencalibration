#pragma once

#include <stddef.h>

#include <eigen3/Eigen/Core>

namespace opencalibration
{

template <typename T>
struct DifferentiableCameraModel
{
    size_t pixels_rows = 0;
    size_t pixels_cols = 0;

    T focal_length_pixels = 0;

    Eigen::Matrix<T,2,1> principle_point {0,0};

    // assumptions: no distortion

    enum class ProjectionType
    {
        PLANAR
    } projection_type = ProjectionType::PLANAR;
};

using CameraModel = DifferentiableCameraModel<double>;
} // namespace opencalibration
