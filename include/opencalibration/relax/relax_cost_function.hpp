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

    PointsDownwardsPrior(double weight) : _weight(weight)
    {
    }

    template <typename T> bool operator()(const T *rotation1, T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using QuaterionTCM = Eigen::Map<const QuaterionT>;

        const QuaterionTCM rotation_em(rotation1);

        const Vector3T cam_center = Eigen::Vector3d(0, 0, 1).cast<T>();
        const Vector3T down = Eigen::Vector3d(0, 0, -1).cast<T>();

        Vector3T rotated_cam_center = rotation_em * cam_center;

        residuals[0] = T(_weight) * angleBetweenUnitVectors<T>(rotated_cam_center, down);
        return true;
    }

  private:
    double _weight;
};

struct DifferenceCost
{
    static const int NUM_RESIDUALS = 1;
    static const int NUM_PARAMETERS_1 = 1;
    static const int NUM_PARAMETERS_2 = 1;

    DifferenceCost(double weight) : _weight(weight)
    {
    }

    template <typename T> bool operator()(const T *val1, const T *val2, T *residual) const
    {
        residual[0] = T(_weight) * (val1[0] - val2[0]);
        return true;
    }

  private:
    const double _weight;
};

struct DistortionMonotonicityCost
{
    static const int NUM_RESIDUALS = 10;
    static const int NUM_PARAMETERS_1 = 3; // k1, k2, k3

    DistortionMonotonicityCost(double r_max, double weight) : _r_max(r_max), _weight(weight)
    {
    }

    template <typename T> bool operator()(const T *radial, T *residuals) const
    {
        // d(r_d)/dr = 1 + 3*k1*r² + 5*k2*r⁴ + 7*k3*r⁶
        // Penalize when this derivative goes negative (non-monotonic)
        for (int i = 0; i < NUM_RESIDUALS; i++)
        {
            T r = T(_r_max * (i + 1.0) / NUM_RESIDUALS);
            T r2 = r * r;
            T r4 = r2 * r2;
            T r6 = r4 * r2;
            T deriv = T(1) + T(3) * radial[0] * r2 + T(5) * radial[1] * r4 + T(7) * radial[2] * r6;
            residuals[i] = deriv < T(0) ? T(_weight) * (-deriv) : T(0);
        }
        return true;
    }

  private:
    double _r_max;
    double _weight;
};

// cost functions for rotations relative to positions
struct DecomposedRotationCost
{
    static const int NUM_RESIDUALS = 3;
    static const int NUM_PARAMETERS_1 = 4;
    static const int NUM_PARAMETERS_2 = 4;

    DecomposedRotationCost(const Eigen::Quaterniond &relative_rotation, const Eigen::Vector3d &relative_translation,
                           const Eigen::Vector3d *translation1, const Eigen::Vector3d *translation2, int score)
        :

          _has_translation((*translation2 - *translation1).squaredNorm() > 1e-9 &&
                           relative_translation.squaredNorm() > 1e-9),
          _relative_rotation(relative_rotation.normalized()),
          _translation_direction((*translation2 - *translation1).normalized()),
          _relative_translation_direction(relative_translation.normalized()), _weight(std::sqrt(score / 8.))
    {
    }

    template <typename T> bool operator()(const T *rotation1, const T *rotation2, T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using QuaterionTCM = Eigen::Map<const QuaterionT>;

        const QuaterionTCM rotation1_em(rotation1);
        const QuaterionTCM rotation2_em(rotation2);

        T res[3];

        if (_has_translation)
        {
            // angle from camera1 -> camera2
            const Vector3T rotated_translation2_1 = rotation1_em.inverse() * _translation_direction.cast<T>();
            res[0] = angleBetweenUnitVectors<T>(rotated_translation2_1, _relative_translation_direction.cast<T>());

            // angle from camera2 -> camera1
            const Vector3T rotated_translation1_2 =
                rotation2_em.inverse() * (_relative_rotation * -_translation_direction).cast<T>();
            res[1] = angleBetweenUnitVectors<T>(rotated_translation1_2, -_relative_translation_direction.cast<T>());
        }
        else
        {
            res[0] = res[1] = T(M_PI);
        }

        // relative orientation of camera1 and camera2
        const QuaterionT rotation2_1 = rotation1_em * rotation2_em.inverse();
        res[2] = Eigen::AngleAxis<T>(_relative_rotation.cast<T>() * rotation2_1).angle();

        for (int i = 0; i < 3; i++)
        {
            residuals[i] = T(_weight) * res[i];
        }

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
    const bool _has_translation;
    const Eigen::Quaterniond _relative_rotation;
    const Eigen::Vector3d _translation_direction, _relative_translation_direction;
    const double _weight;
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
        int max_score = 0;
        for (const auto &pose : relations.relative_poses)
        {
            if (pose.score > max_score)
                max_score = pose.score;
        }

        for (const auto &pose : relations.relative_poses)
        {
            if (pose.score > 0.25 * max_score)
            {
                decompose.emplace_back(pose.orientation, pose.position, translation1, translation2, pose.score);
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
        for (const auto &d : decompose)
        {
            VectorRT res;
            if (d(rotation1, rotation2, res.data()) && res.allFinite() && res.squaredNorm() < lowest_res_norm)
            {
                lowest_res_norm = res.squaredNorm();
                lowest_res = res;
            }
        }

        VectorRTM res_vec(residuals);
        res_vec = lowest_res;

        return ceres::isfinite(lowest_res_norm);
    }

    std::vector<DecomposedRotationCost> decompose;
};

struct PixelErrorCost_Orientation
{
    static const int NUM_RESIDUALS = 2;
    static const int NUM_PARAMETERS_1 = 4;
    static const int NUM_PARAMETERS_2 = 3;

    PixelErrorCost_Orientation(const Eigen::Vector3d &camera_loc, const CameraModel &camera_model,
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

        const Vector2T projected_pixel = image_from_3d<T>(point_em, model.cast<T>(), loc.cast<T>(), rotation_em);

        Vector2TM residuals_m(residuals);
        residuals_m = projected_pixel - pixel.cast<T>();

        return true;
    }

  private:
    const Eigen::Vector3d &loc;
    const CameraModel &model;
    const Eigen::Vector2d pixel; // unique to this measurement, keep a local copy to avoid cache thrashing
};

struct PixelErrorCost_OrientationFocal
{
    static const int NUM_RESIDUALS = 2;
    static const int NUM_PARAMETERS_1 = 4;
    static const int NUM_PARAMETERS_2 = 3;
    static const int NUM_PARAMETERS_3 = 1;
    static const int NUM_PARAMETERS_4 = 2;

    PixelErrorCost_OrientationFocal(const Eigen::Vector3d &camera_loc, const CameraModel &camera_model,
                                    const Eigen::Vector2d &camera_pixel)
        : loc(camera_loc), model(camera_model), pixel(camera_pixel)
    {
    }

    template <typename T>
    bool operator()(const T *rotation, const T *point, const T *focal, const T *principal, T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using Vector2T = Eigen::Matrix<T, 2, 1>;
        using QuaterionTCM = Eigen::Map<const QuaterionT>;
        using Vector3TCM = Eigen::Map<const Vector3T>;
        using Vector2TCM = Eigen::Map<const Vector2T>;
        using Vector2TM = Eigen::Map<Vector2T>;

        const QuaterionTCM rotation_em(rotation);
        const Vector3TCM point_em(point);

        Vector3T ray = rotation_em.inverse() * (point_em - loc.cast<T>());

        DifferentiableCameraModel<T> model_t = model.cast<T>();
        model_t.focal_length_pixels = *focal;
        model_t.principle_point = Vector2TCM(principal);

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

struct PixelErrorCost_OrientationFocalRadial
{
    static const int NUM_RESIDUALS = 2;
    static const int NUM_PARAMETERS_1 = 4;
    static const int NUM_PARAMETERS_2 = 3;
    static const int NUM_PARAMETERS_3 = 1;
    static const int NUM_PARAMETERS_4 = 2;
    static const int NUM_PARAMETERS_5 = 3;

    PixelErrorCost_OrientationFocalRadial(const Eigen::Vector3d &camera_loc, const CameraModel &camera_model,
                                          const Eigen::Vector2d &camera_pixel)
        : loc(camera_loc), model(camera_model), pixel(camera_pixel)
    {
    }

    template <typename T>
    bool operator()(const T *rotation, const T *point, const T *focal, const T *principal, const T *radial,
                    T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using Vector2T = Eigen::Matrix<T, 2, 1>;
        using QuaterionTCM = Eigen::Map<const QuaterionT>;
        using Vector3TCM = Eigen::Map<const Vector3T>;
        using Vector3TCM_const = Eigen::Map<const Eigen::Matrix<T, 3, 1>>;
        using Vector2TCM = Eigen::Map<const Vector2T>;
        using Vector2TM = Eigen::Map<Vector2T>;

        const QuaterionTCM rotation_em(rotation);
        const Vector3TCM point_em(point);

        Vector3T ray = rotation_em.inverse() * (point_em - loc.cast<T>());

        DifferentiableCameraModel<T> model_t = model.cast<T>();
        model_t.focal_length_pixels = *focal;
        model_t.principle_point = Vector2TCM(principal);
        model_t.radial_distortion = Vector3TCM_const(radial);

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

struct PixelErrorCost_OrientationFocalRadialTangential
{
    static const int NUM_RESIDUALS = 2;
    static const int NUM_PARAMETERS_1 = 4;
    static const int NUM_PARAMETERS_2 = 3;
    static const int NUM_PARAMETERS_3 = 1;
    static const int NUM_PARAMETERS_4 = 2;
    static const int NUM_PARAMETERS_5 = 3;
    static const int NUM_PARAMETERS_6 = 2;

    PixelErrorCost_OrientationFocalRadialTangential(const Eigen::Vector3d &camera_loc, const CameraModel &camera_model,
                                                    const Eigen::Vector2d &camera_pixel)
        : loc(camera_loc), model(camera_model), pixel(camera_pixel)
    {
    }

    template <typename T>
    bool operator()(const T *rotation, const T *point, const T *focal, const T *principal, const T *radial,
                    const T *tangential, T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using Vector2T = Eigen::Matrix<T, 2, 1>;
        using QuaterionTCM = Eigen::Map<const QuaterionT>;
        using Vector3TCM = Eigen::Map<const Vector3T>;
        using Vector3TCM_const = Eigen::Map<const Eigen::Matrix<T, 3, 1>>;
        using Vector2TCM = Eigen::Map<const Vector2T>;
        using Vector2TM = Eigen::Map<Vector2T>;

        const QuaterionTCM rotation_em(rotation);
        const Vector3TCM point_em(point);

        Vector3T ray = rotation_em.inverse() * (point_em - loc.cast<T>());

        DifferentiableCameraModel<T> model_t = model.cast<T>();
        model_t.focal_length_pixels = *focal;
        model_t.principle_point = Vector2TCM(principal);
        model_t.radial_distortion = Vector3TCM_const(radial);
        model_t.tangential_distortion = Vector2TCM(tangential);

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

struct PlaneIntersectionAngleCost_OrientationFocalRadial_SharedModel
{
    static const int NUM_RESIDUALS = 3;
    static const int NUM_PARAMETERS_1 = 4; // rotation 0
    static const int NUM_PARAMETERS_2 = 4; // rotation 1
    static const int NUM_PARAMETERS_3 = 1; // z 0
    static const int NUM_PARAMETERS_4 = 1; // z 1
    static const int NUM_PARAMETERS_5 = 1; // z 2
    static const int NUM_PARAMETERS_6 = 1; // focal
    static const int NUM_PARAMETERS_7 = 2; // principal
    static const int NUM_PARAMETERS_8 = 3; // radial

    PlaneIntersectionAngleCost_OrientationFocalRadial_SharedModel(
        const Eigen::Vector3d &camera_loc1, const Eigen::Vector3d &camera_loc2, const Eigen::Vector2d &camera_pixel1,
        const Eigen::Vector2d &camera_pixel2, const Eigen::Vector2d &plane_point1, const Eigen::Vector2d &plane_point2,
        const Eigen::Vector2d &plane_point3, const InverseDifferentiableCameraModel<double> &sharedModel)
        : camera_loc{camera_loc1, camera_loc2}, camera_pixel{camera_pixel1, camera_pixel2}, plane_point{plane_point1,
                                                                                                        plane_point2,
                                                                                                        plane_point3},
          sharedModel(sharedModel)
    {
    }

    template <typename T>
    bool operator()(const T *rotation0, const T *rotation1, const T *z0, const T *z1, const T *z2, const T *focal,
                    const T *principal, const T *radial, T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using Vector2T = Eigen::Matrix<T, 2, 1>;
        using QuaternionTCM = Eigen::Map<const QuaterionT>;
        using Vector3TM = Eigen::Map<Vector3T>;
        using Vector3TCM = Eigen::Map<const Vector3T>;
        using Vector2TCM = Eigen::Map<const Vector2T>;

        const QuaternionTCM rotation_em[2]{QuaternionTCM(rotation0), QuaternionTCM(rotation1)};
        InverseDifferentiableCameraModel<T> model = sharedModel.cast<T>();
        model.focal_length_pixels = T(*focal);
        model.principle_point = Vector2TCM(principal);
        model.radial_distortion = Vector3TCM(radial);
        ray<T> rays[2];
        for (int i = 0; i < 2; i++)
        {
            rays[i].dir = rotation_em[i] * image_to_3d<T>(camera_pixel[i].cast<T>(), model);
            rays[i].offset = camera_loc[i].cast<T>();
        }

        const T plane_z[3]{*z0, *z1, *z2};
        plane_3_corners<T> plane3;
        for (int i = 0; i < 3; i++)
            plane3.corner[i] << T(plane_point[i].x()), T(plane_point[i].y()), plane_z[i];

        plane_norm_offset<T> pno = cornerPlane2normOffsetPlane(plane3);

        Vector3T intersection[2];
        bool intersections = true;
        for (int i = 0; i < 2; i++)
            intersections &= rayPlaneIntersection(rays[i], pno, intersection[i]);

        Vector3T distance_error = intersection[0] - intersection[1];
        T avg_dist = ((intersection[0] - camera_loc[0]).norm() + (intersection[1] - camera_loc[1]).norm()) * T(0.5);
        Vector3T angle_error = distance_error / avg_dist;

        Vector3TM final_error(residuals);
        final_error = angle_error;

        return intersections;
    }

  private:
    const std::array<Eigen::Vector3d, 2> camera_loc;
    const std::array<Eigen::Vector2d, 2> camera_pixel;
    const std::array<Eigen::Vector2d, 3> plane_point;
    const InverseDifferentiableCameraModel<double> sharedModel;
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
        bool intersections = true;
        for (int i = 0; i < 2; i++)
            intersections &= rayPlaneIntersection(rays[i], pno, intersection[i]);

        Vector3T distance_error = intersection[0] - intersection[1];
        T avg_dist = ((intersection[0] - camera_loc[0]).norm() + (intersection[1] - camera_loc[1]).norm()) * T(0.5);
        Vector3T angle_error = distance_error / avg_dist;

        Vector3TM final_error(residuals);
        final_error = angle_error;

        return intersections;
    }

  private:
    const std::array<Eigen::Vector3d, 2> camera_loc;
    const std::array<Eigen::Vector3d, 2> camera_ray;
    const std::array<Eigen::Vector2d, 3> plane_point;
};

} // namespace opencalibration
