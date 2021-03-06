#pragma once

#include <opencalibration/types/camera_model.hpp>
#include <opencalibration/types/correspondence.hpp>
#include <opencalibration/types/feature_2d.hpp>
#include <opencalibration/types/feature_match.hpp>

#include <eigen3/Eigen/Geometry>

#include <vector>

namespace opencalibration
{

std::vector<correspondence> distort_keypoints(const std::vector<feature_2d> &features1,
                                              const std::vector<feature_2d> &features2,
                                              const std::vector<feature_match> &matches, const CameraModel &model1,
                                              const CameraModel &model2);

Eigen::Vector3d image_to_3d(const Eigen::Vector2d &keypoint, const CameraModel &model);

// should be differentiable
template <typename T = double>
Eigen::Matrix<T, 2, 1> image_from_3d(const Eigen::Matrix<T, 3, 1> &ray, const DifferentiableCameraModel<T> &model)
{
    Eigen::Matrix<T, 2, 1> pixel_location;
    switch (model.projection_type)
    {
    case ProjectionType::PLANAR: {
        pixel_location = ray.hnormalized() * model.focal_length_pixels + model.principle_point;
        break;
    }
    case ProjectionType::UNKNOWN:
        pixel_location.fill(T(NAN));
        break;
    }
    return pixel_location;
}

} // namespace opencalibration
