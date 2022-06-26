#pragma once

#include <opencalibration/types/camera_model.hpp>
#include <opencalibration/types/correspondence.hpp>
#include <opencalibration/types/feature_2d.hpp>
#include <opencalibration/types/feature_match.hpp>
#include <opencalibration/types/ray.hpp>

#include <eigen3/Eigen/Geometry>

#include <vector>

namespace opencalibration
{

std::vector<correspondence> distort_keypoints(const std::vector<feature_2d> &features1,
                                              const std::vector<feature_2d> &features2,
                                              const std::vector<feature_match> &matches, const CameraModel &model1,
                                              const CameraModel &model2);

Eigen::Vector3d image_to_3d(const Eigen::Vector2d &keypoint, const CameraModel &model);

template <typename T = double>
Eigen::Matrix<T, 2, 1> distortProjectedRay(const Eigen::Matrix<T, 2, 1> &ray_projected,
                                           const DifferentiableCameraModel<T> &model)
{

    Eigen::Matrix<T, 3, 1> r2;
    r2[0] = ray_projected.squaredNorm();
    for (Eigen::Index i = 1; i < r2.size(); i++)
        r2[i] = r2[i - 1] * r2[0];

    const Eigen::Matrix<T, 2, 1> ray_distorted =
        (T(1) + model.radial_distortion.dot(r2)) * ray_projected.array() +
        T(2) * ray_projected.prod() * model.tangential_distortion.array() +
        model.tangential_distortion.reverse().array() * (r2[0] + T(2) * ray_projected.array() * ray_projected.array());

    return ray_distorted;
}

template <typename T = double>
Eigen::Matrix<T, 2, 1> image_from_3d(const Eigen::Matrix<T, 3, 1> &ray, const DifferentiableCameraModel<T> &model)
{
    Eigen::Matrix<T, 2, 1> ray_projected;
    switch (model.projection_type)
    {
    case ProjectionType::PLANAR: {
        ray_projected = ray.hnormalized();
        break;
    }
    case ProjectionType::UNKNOWN:
        ray_projected.fill(T(NAN));
        break;
    }

    const Eigen::Matrix<T, 2, 1> ray_distorted = distortProjectedRay<T>(ray_projected, model);

    Eigen::Matrix<T, 2, 1> pixel_location = ray_distorted * model.focal_length_pixels + model.principle_point;
    return pixel_location;
}

template <typename T = double>
Eigen::Matrix<T, 2, 1> image_from_3d(const Eigen::Matrix<T, 3, 1> &point, const DifferentiableCameraModel<T> &model,
                                     const Eigen::Matrix<T, 3, 1> &camera_location,
                                     const Eigen::Quaternion<T> &camera_orientation)
{
    auto ray = camera_location - point;
    auto rotated_ray = camera_orientation * ray;
    return image_from_3d(rotated_ray, model);
}

} // namespace opencalibration
