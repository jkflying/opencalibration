#pragma once

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geometry/intersection.hpp>

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>

#include <unordered_set>

namespace opencalibration
{

template <typename T> T angleBetweenUnitVectors(const Eigen::Matrix<T, 3, 1> &n1, const Eigen::Matrix<T, 3, 1> &n2)
{
    return acos(T(0.99999) * n1.dot(n2));
}

struct PointsDownwardsPrior
{
    static constexpr double weight = 1e-3;
    template <typename T> bool operator()(const T *rotation1, T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using QuaterionTCM = Eigen::Map<const QuaterionT>;

        const QuaterionTCM rotation_em(rotation1);

        const Vector3T cam_center = Eigen::Vector3d(0, 0, 1).cast<T>();
        const Vector3T down = Eigen::Vector3d(0, 0, -1).cast<T>();

        Vector3T rotated_cam_center = rotation_em * cam_center;

        residuals[0] = weight * angleBetweenUnitVectors<T>(rotated_cam_center, down);
        return true;
    }
};

// cost functions for rotations relative to positions

struct DecomposedRotationCost
{
    DecomposedRotationCost(const Eigen::Quaterniond &relative_rotation, const Eigen::Vector3d &relative_translation,
                           const Eigen::Vector3d *translation1, const Eigen::Vector3d *translation2)
        :

          _has_translation((*translation2 - *translation1).squaredNorm() > 1e-9 &&
                           relative_translation.squaredNorm() > 1e-9),
          _relative_rotation(relative_rotation.normalized()),
          _translation_direction((*translation2 - *translation1).normalized()),
          _relative_translation_direction(relative_translation.normalized())
    {
    }

    template <typename T> bool operator()(const T *rotation1, const T *rotation2, T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using QuaterionTCM = Eigen::Map<const QuaterionT>;

        const QuaterionTCM rotation1_em(rotation1);
        const QuaterionTCM rotation2_em(rotation2);

        if (_has_translation)
        {
            // angle from camera1 -> camera2
            const Vector3T rotated_translation2_1 = rotation1_em.inverse() * _translation_direction.cast<T>();
            residuals[0] =
                angleBetweenUnitVectors<T>(rotated_translation2_1, _relative_translation_direction.cast<T>());

            // angle from camera2 -> camera1
            const Vector3T rotated_translation1_2 =
                rotation2_em.inverse() * (_relative_rotation * -_translation_direction).cast<T>();
            residuals[1] =
                angleBetweenUnitVectors<T>(rotated_translation1_2, -_relative_translation_direction.cast<T>());
        }
        else
        {
            residuals[0] = residuals[1] = T(M_PI);
        }

        // relative orientation of camera1 and camera2
        const QuaterionT rotation2_1 = rotation1_em.inverse() * rotation2_em;
        residuals[2] = Eigen::AngleAxis<T>(_relative_rotation.cast<T>() * rotation2_1).angle();

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
    const bool _has_translation;
    const Eigen::Quaterniond _relative_rotation;
    const Eigen::Vector3d _translation_direction, _relative_translation_direction;
};

// Homographies give two valid decompositions. Use this to evaluate the cost of these paired decompositions.
// Takes the residual from the decomposition with lower error, or whichever doesn't have NaN/Inf in it
struct DualDecomposedRotationCost
{
    DualDecomposedRotationCost(const camera_relations &relations, const Eigen::Vector3d *translation1,
                               const Eigen::Vector3d *translation2)
        : decompose1(relations.relative_rotation, relations.relative_translation, translation1, translation2),
          decompose2(relations.relative_rotation2, relations.relative_translation2, translation1, translation2)
    {
    }

    template <typename T> bool operator()(const T *rotation1, const T *rotation2, T *residuals) const
    {
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using Vector3TM = Eigen::Map<Vector3T>;

        Vector3T res[2];
        bool success1 = decompose1(rotation1, rotation2, res[0].data()) && res[0].allFinite();
        bool success2 = decompose2(rotation1, rotation2, res[1].data()) && res[1].allFinite();

        Vector3TM res_vec(residuals);
        if (success1 && success2)
        {
            res_vec = (res[0].squaredNorm() <= res[1].squaredNorm()) ? res[0] : res[1];
        }
        else if (success1)
        {
            res_vec = res[0];
        }
        else if (success2)
        {
            res_vec = res[1];
        }

        return success1 || success2;
    }

    DecomposedRotationCost decompose1, decompose2;
};

struct PixelErrorCost
{
    PixelErrorCost(const Eigen::Vector3d &camera_loc, const CameraModel &camera_model,
                   const Eigen::Vector2d &camera_pixel)
        : loc(camera_loc), model(camera_model), pixel(camera_pixel)
    {
    }

    template <typename T> bool operator()(const T *rotation, const T *point, T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using Vector2T = Eigen::Matrix<T, 2, 1>;
        using QuaterionTCM = Eigen::Map<const QuaterionT>;
        using Vector3TCM = Eigen::Map<const Vector3T>;
        using Vector2TM = Eigen::Map<Vector2T>;

        const QuaterionTCM rotation_em(rotation);
        const Vector3TCM point_em(point);

        Vector3T ray = rotation_em.inverse() * (loc.cast<T>() - point_em);

        DifferentiableCameraModel<T> model_t = model.cast<T>();

        Vector2T projected_pixel = image_from_3d<T>(ray, model_t);

        Vector2TM residuals_m(residuals);
        residuals_m = projected_pixel - pixel.cast<T>();

        return true;
    }

  private:
    const Eigen::Vector3d &loc;
    const CameraModel &model;
    const Eigen::Vector2d &pixel;
};
} // namespace opencalibration
