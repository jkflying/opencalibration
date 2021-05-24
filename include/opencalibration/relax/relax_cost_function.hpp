#pragma once

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geometry/intersection.hpp>

#include <opencalibration/types/plane.hpp>

#include <ceres/jet.h>
#include <spdlog/spdlog.h>

#include <unordered_set>

namespace opencalibration
{

template <typename T> T angleBetweenUnitVectors(const Eigen::Matrix<T, 3, 1> &n1, const Eigen::Matrix<T, 3, 1> &n2)
{
    return acos(std::clamp<T>(n1.dot(n2), T(-1 + 1e-12), T(1 - 1e-12)));
}

struct PointsDownwardsPrior
{
    static const int NUM_RESIDUALS = 1;
    static const int NUM_PARAMETERS_1 = 4;

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
    static const int NUM_RESIDUALS = 3;
    static const int NUM_PARAMETERS_1 = 4;
    static const int NUM_PARAMETERS_2 = 4;

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
        const QuaterionT rotation2_1 = rotation1_em * rotation2_em.inverse();
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
struct MultiDecomposedRotationCost
{
    static const int NUM_RESIDUALS = DecomposedRotationCost::NUM_RESIDUALS;
    static const int NUM_PARAMETERS_1 = 4;
    static const int NUM_PARAMETERS_2 = 4;

    MultiDecomposedRotationCost(const camera_relations &relations, const Eigen::Vector3d *translation1,
                                const Eigen::Vector3d *translation2)
    {
        decompose.reserve(relations.relative_poses.size());
        for (const auto &pose : relations.relative_poses)
        {
            if (pose.score >= 0)
            {
                decompose.emplace_back(pose.orientation, pose.position, translation1, translation2);
            }
        }
    }

    template <typename T> bool operator()(const T *rotation1, const T *rotation2, T *residuals) const
    {
        using VectorRT = Eigen::Matrix<T, NUM_RESIDUALS, 1>;
        using VectorRTM = Eigen::Map<VectorRT>;

        T lowest_res_norm(std::numeric_limits<double>::infinity());
        VectorRT lowest_res;
        lowest_res.fill(T(NAN));
        for (size_t i = 0; i < decompose.size(); i++)
        {
            VectorRT res;
            if (decompose[i](rotation1, rotation2, res.data()) && res.allFinite() &&
                res.squaredNorm() < lowest_res_norm)
            {
                lowest_res_norm = res.squaredNorm();
                lowest_res = res;
            }
        }

        VectorRTM res_vec(residuals);
        res_vec = lowest_res;

        return ceres::IsFinite(lowest_res_norm);
    }

    std::vector<DecomposedRotationCost> decompose;
};

struct PixelErrorCost
{
    static const int NUM_RESIDUALS = 2;
    static const int NUM_PARAMETERS_1 = 4;
    static const int NUM_PARAMETERS_2 = 3;

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
    const Eigen::Vector2d pixel; // unique to this measurement, keep a local copy to avoid cache thrashing
};

struct PlaneIntersectionAngleCost
{
    static const int NUM_RESIDUALS = 3;
    static const int NUM_PARAMETERS_1 = 4;
    static const int NUM_PARAMETERS_2 = 4;
    static const int NUM_PARAMETERS_3 = 1;
    static const int NUM_PARAMETERS_4 = 1;
    static const int NUM_PARAMETERS_5 = 1;

    PlaneIntersectionAngleCost(const Eigen::Vector3d &camera_loc1, const Eigen::Vector3d &camera_loc2,
                               const Eigen::Vector3d &camera_ray1, const Eigen::Vector3d &camera_ray2,
                               const Eigen::Vector2d &plane_point1, const Eigen::Vector2d &plane_point2,
                               const Eigen::Vector2d &plane_point3)
        : camera_loc{camera_loc1, camera_loc2}, camera_ray{camera_ray1, camera_ray2}, plane_point{plane_point1,
                                                                                                  plane_point2,
                                                                                                  plane_point3}
    {
    }

    template <typename T>
    bool operator()(const T *rotation0, const T *rotation1, const T *z0, const T *z1, const T *z2, T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using QuaternionTCM = Eigen::Map<const QuaterionT>;
        using Vector3TM = Eigen::Map<Vector3T>;

        const QuaternionTCM rotation_em[2]{QuaternionTCM(rotation0), QuaternionTCM(rotation1)};
        ray<T> rays[2];
        for (int i = 0; i < 2; i++)
        {
            rays[i].dir = rotation_em[i] * camera_ray[i].cast<T>();
            rays[i].offset = camera_loc[i].cast<T>();
        }

        const T plane_z[3]{*z0, *z1, *z2};
        plane_3_corners<T> plane3;
        for (int i = 0; i < 3; i++)
            plane3.corner[i] << T(plane_point[i].x()), T(plane_point[i].y()), plane_z[i];

        plane_norm_offset<T> pno = cornerPlane2normOffsetPlane(plane3);

        Vector3T intersection[2];
        for (int i = 0; i < 2; i++)
            intersection[i] = rayPlaneIntersection(rays[i], pno);

        Vector3T distance_error = intersection[0] - intersection[1];
        T avg_dist = ((intersection[0] - camera_loc[0]).norm() + (intersection[1] - camera_loc[1]).norm()) * T(0.5);
        Vector3T angle_error = distance_error / avg_dist;

        Vector3TM final_error(residuals);
        final_error = angle_error;

        return true;
    }

  private:
    const Eigen::Vector3d camera_loc[2];
    const Eigen::Vector3d camera_ray[2];
    const Eigen::Vector2d plane_point[3];
};

} // namespace opencalibration
