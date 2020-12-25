#pragma once

#include <stddef.h>

#include <eigen3/Eigen/Core>

namespace opencalibration
{

template <typename T> struct DifferentiableCameraModel
{
    size_t pixels_rows = 0;
    size_t pixels_cols = 0;

    T focal_length_pixels = 0;

    Eigen::Matrix<T, 2, 1> principle_point{0, 0};

    // assumptions: no distortion

    enum class ProjectionType
    {
        PLANAR,
        UNKNOWN
    } projection_type = ProjectionType::PLANAR;

    bool operator==(const DifferentiableCameraModel &other) const
    {
        return pixels_rows == other.pixels_rows && pixels_cols == other.pixels_cols &&
               focal_length_pixels == other.focal_length_pixels && principle_point == other.principle_point &&
               projection_type == other.projection_type;
    }
};

using CameraModel = DifferentiableCameraModel<double>;
} // namespace opencalibration
