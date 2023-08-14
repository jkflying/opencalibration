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

enum class CameraModelTag
{
    FORWARD,
    INVERSE
};

template <typename T, CameraModelTag tag> struct DifferentiableCameraModelBase
{
    static const CameraModelTag tagValue = tag;

    size_t pixels_rows = 0;
    size_t pixels_cols = 0;

    T focal_length_pixels = T(0);

    Eigen::Matrix<T, 2, 1> principle_point{T(0), T(0)};

    Eigen::Matrix<T, 3, 1> radial_distortion = Eigen::Matrix<T, 3, 1>::Zero();
    Eigen::Matrix<T, 2, 1> tangential_distortion = Eigen::Matrix<T, 2, 1>::Zero();

    ProjectionType projection_type = ProjectionType::PLANAR;

    bool operator==(const DifferentiableCameraModelBase &other) const
    {
        return pixels_rows == other.pixels_rows && pixels_cols == other.pixels_cols &&
               focal_length_pixels == other.focal_length_pixels && principle_point == other.principle_point &&
               radial_distortion == other.radial_distortion && tangential_distortion == other.tangential_distortion &&
               projection_type == other.projection_type;
    }

    template <typename K, CameraModelTag result_tag = tag> DifferentiableCameraModelBase<K, result_tag> cast() const
    {
        DifferentiableCameraModelBase<K, result_tag> model;

        model.pixels_rows = pixels_rows;
        model.pixels_cols = pixels_cols;
        model.focal_length_pixels = K(focal_length_pixels);
        model.principle_point = principle_point.template cast<K>();
        model.radial_distortion = radial_distortion.template cast<K>();
        model.tangential_distortion = tangential_distortion.template cast<K>();
        model.projection_type = projection_type;

        return model;
    }
};

template <typename T> using DifferentiableCameraModel = DifferentiableCameraModelBase<T, CameraModelTag::FORWARD>;

template <typename T>
using InverseDifferentiableCameraModel = DifferentiableCameraModelBase<T, CameraModelTag::INVERSE>;

struct CameraModel final : public DifferentiableCameraModel<double>
{
  public:
    size_t id = 0;

    bool operator==(const CameraModel &other) const
    {
        return id == other.id && DifferentiableCameraModel<double>::operator==(other);
    }
};

struct InverseCameraModel final : public InverseDifferentiableCameraModel<double>
{
  public:
    size_t id = 0;

    bool operator==(const InverseCameraModel &other) const
    {
        return id == other.id && InverseDifferentiableCameraModel<double>::operator==(other);
    }
};

} // namespace opencalibration
