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
                                              const std::vector<feature_match> &matches,
                                              const DifferentiableCameraModel<double> &model1,
                                              const DifferentiableCameraModel<double> &model2);

Eigen::Vector3d image_to_3d(const Eigen::Vector2d &keypoint, const DifferentiableCameraModel<double> &model);
Eigen::Vector2d image_from_3d(const Eigen::Vector3d &ray, const InverseDifferentiableCameraModel<double> &model);

template <typename T>
Eigen::Matrix<T, 2, 1> distortProjectedRay(const Eigen::Matrix<T, 2, 1> &ray_projected,
                                           const Eigen::Matrix<T, 3, 1> &radial_distortion,
                                           const Eigen::Matrix<T, 2, 1> &tangential_distortion)
{

    Eigen::Matrix<T, 3, 1> r2;
    r2[0] = ray_projected.squaredNorm();
    for (Eigen::Index i = 1; i < r2.size(); i++)
        r2[i] = r2[i - 1] * r2[0];

    const Eigen::Matrix<T, 2, 1> ray_distorted =
        (T(1) + radial_distortion.dot(r2)) * ray_projected.array() +
        T(2) * ray_projected.prod() * tangential_distortion.array() +
        tangential_distortion.reverse().array() * (r2[0] + T(2) * ray_projected.array() * ray_projected.array());

    return ray_distorted;
}

template <typename T>
Eigen::Matrix<T, 2, 1> image_from_3d(const Eigen::Matrix<T, 3, 1> &ray, const DifferentiableCameraModel<T> &model)
{
    Eigen::Matrix<T, 2, 1> ray_projected;
    switch (model.projection_type)
    {
    case ProjectionType::PLANAR: {
        const T z = ray.z();
        const T min_z = T(1e-3);
        const T clamped_z = (z < min_z) ? min_z : z;
        ray_projected = ray.template head<2>() / clamped_z;
        break;
    }
    case ProjectionType::UNKNOWN:
        ray_projected.fill(T(NAN));
        break;
    }

    const Eigen::Matrix<T, 2, 1> ray_distorted =
        distortProjectedRay<T>(ray_projected, model.radial_distortion, model.tangential_distortion);

    const Eigen::Matrix<T, 2, 1> pixel_location = ray_distorted * model.focal_length_pixels + model.principle_point;
    return pixel_location;
}

template <typename T = double>
Eigen::Matrix<T, 2, 1> image_from_3d(const Eigen::Matrix<T, 3, 1> &point, const DifferentiableCameraModel<T> &model,
                                     const Eigen::Matrix<T, 3, 1> &camera_location,
                                     const Eigen::Quaternion<T> &camera_orientation)
{
    const Eigen::Matrix<T, 3, 1> rotated_ray = camera_orientation.inverse() * (point - camera_location);
    return image_from_3d(rotated_ray, model);
}

// Overload accepting precomputed inverse rotation matrix (avoids quaternion.inverse() per call)
template <typename T = double>
Eigen::Matrix<T, 2, 1> image_from_3d(const Eigen::Matrix<T, 3, 1> &point, const DifferentiableCameraModel<T> &model,
                                     const Eigen::Matrix<T, 3, 1> &camera_location,
                                     const Eigen::Matrix<T, 3, 3> &camera_orientation_inverse)
{
    const Eigen::Matrix<T, 3, 1> rotated_ray = camera_orientation_inverse * (point - camera_location);
    return image_from_3d(rotated_ray, model);
}

inline Eigen::Vector2d image_from_3d(const Eigen::Vector3d &point,
                                     const InverseDifferentiableCameraModel<double> &model,
                                     const Eigen::Vector3d &camera_location,
                                     const Eigen::Quaterniond &camera_orientation)
{
    const Eigen::Vector3d rotated_ray = camera_orientation.inverse() * (point - camera_location);
    return image_from_3d(rotated_ray, model);
}

template <typename T = double>
Eigen::Matrix<T, 3, 1> image_to_3d(const Eigen::Matrix<T, 2, 1> &keypoint,
                                   const InverseDifferentiableCameraModel<T> &model)
{
    const Eigen::Matrix<T, 2, 1> unprojected_point = (keypoint - model.principle_point) / model.focal_length_pixels;

    const Eigen::Matrix<T, 2, 1> undistorted_point =
        distortProjectedRay<T>(unprojected_point, model.radial_distortion, model.tangential_distortion);

    Eigen::Matrix<T, 3, 1> ray;
    switch (model.projection_type)
    {
    case ProjectionType::PLANAR:
        ray = undistorted_point.homogeneous().normalized();
        break;
    case ProjectionType::UNKNOWN:
        break;
    }
    return ray;
}

template <typename T = double>
ray<T> image_to_3d(const Eigen::Matrix<T, 2, 1> &keypoint, const InverseDifferentiableCameraModel<T> &model,
                   const Eigen::Matrix<T, 3, 1> &camera_location, const Eigen::Quaternion<T> &camera_orientation)
{
    return ray<T>{camera_orientation * image_to_3d(keypoint, model), camera_location};
}

inline ray_d image_to_3d(const Eigen::Vector2d &keypoint, const DifferentiableCameraModel<double> &model,
                         const Eigen::Vector3d &camera_location, const Eigen::Quaterniond &camera_orientation)
{
    return ray_d{camera_orientation * image_to_3d(keypoint, model), camera_location};
}

} // namespace opencalibration
