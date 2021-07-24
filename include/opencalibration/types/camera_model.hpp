#pragma once

#include <stddef.h>

#include <eigen3/Eigen/Core>

namespace opencalibration
{

enum class ProjectionType
{
    PLANAR,
    UNKNOWN
};

template <typename T> struct DifferentiableCameraModel
{
    size_t pixels_rows = 0;
    size_t pixels_cols = 0;

    T focal_length_pixels = T(0);

    Eigen::Matrix<T, 2, 1> principle_point{T(0), T(0)};

    // TODO: add distortion

    ProjectionType projection_type = ProjectionType::PLANAR;

    bool operator==(const DifferentiableCameraModel &other) const
    {
        return pixels_rows == other.pixels_rows && pixels_cols == other.pixels_cols &&
               focal_length_pixels == other.focal_length_pixels && principle_point == other.principle_point &&
               projection_type == other.projection_type;
    }

    template <typename K> DifferentiableCameraModel<K> cast() const
    {
        DifferentiableCameraModel<K> model;

        model.pixels_rows = pixels_rows;
        model.pixels_cols = pixels_cols;
        model.focal_length_pixels = K(focal_length_pixels);
        model.principle_point = principle_point.template cast<K>();
        model.projection_type = projection_type;

        return model;
    }
};

class CameraModel final : public DifferentiableCameraModel<double>
{
  public:
    size_t id = 0;

    bool operator==(const CameraModel &other)
    {
        return id == other.id && DifferentiableCameraModel<double>::operator==(other);
    }
};
} // namespace opencalibration
