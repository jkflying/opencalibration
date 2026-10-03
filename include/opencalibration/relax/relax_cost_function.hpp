#pragma once

#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geometry/intersection.hpp>
#include <opencalibration/types/camera_relations.hpp>
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

template <typename T>
T signedDihedralAngle(const Eigen::Matrix<T, 3, 1> &A, const Eigen::Matrix<T, 3, 1> &B, const Eigen::Matrix<T, 3, 1> &C,
                      const Eigen::Matrix<T, 3, 1> &D)
{
    const Eigen::Matrix<T, 3, 1> AB = B - A;
    const Eigen::Matrix<T, 3, 1> n1 = AB.cross(C - A);
    const Eigen::Matrix<T, 3, 1> n2 = (D - A).cross(AB);
    return atan2(n1.cross(n2).dot(AB), n1.dot(n2) * AB.norm());
}

static constexpr int POSE_PARAMETERS = 7;
static constexpr int ORIENTATION_PARAMETERS = 4;

constexpr double RAY_PIXEL_SIGMA = 1.0;

template <int N> std::array<double, N> unitSigmas()
{
    std::array<double, N> a;
    a.fill(1);
    return a;
}

template <int N, typename T> std::array<Eigen::Matrix<T, 3, 1>, N> posePositions(const T *const *poses)
{
    std::array<Eigen::Matrix<T, 3, 1>, N> positions;
    for (int i = 0; i < N; i++)
        positions[i] = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(poses[i] + 4);
    return positions;
}

template <typename T, size_t N>
std::array<Eigen::Matrix<T, 3, 1>, N> castPositions(const std::array<Eigen::Vector3d, N> &positions)
{
    std::array<Eigen::Matrix<T, 3, 1>, N> cast;
    for (size_t i = 0; i < N; i++)
        cast[i] = positions[i].template cast<T>();
    return cast;
}

template <typename T> T pointsDownwardsResidual(const T *rotation, double weight)
{
    using Vector3T = Eigen::Matrix<T, 3, 1>;

    const Eigen::Map<const Eigen::Quaternion<T>> rotation_em(rotation);

    const Vector3T cam_center = Eigen::Vector3d(0, 0, 1).cast<T>();
    const Vector3T down = Eigen::Vector3d(0, 0, -1).cast<T>();

    Vector3T rotated_cam_center = rotation_em * cam_center;

    constexpr double MAX_TILT = M_PI / 4, BEYOND_MAX_TILT_SCALE = 1000;
    const T angle = angleBetweenUnitVectors<T>(rotated_cam_center, down);
    T residual = T(weight) * angle;
    if (angle > T(MAX_TILT))
        residual += T(weight * BEYOND_MAX_TILT_SCALE) * (angle - T(MAX_TILT));
    return residual;
}

struct PointsDownwardsPrior
{
    static const int NUM_RESIDUALS = 1;
    static const int NUM_PARAMETERS_1 = POSE_PARAMETERS;

    PointsDownwardsPrior(double weight) : _weight(weight)
    {
    }

    template <typename T> bool operator()(const T *pose1, T *residuals) const
    {
        residuals[0] = pointsDownwardsResidual(pose1, _weight);
        return true;
    }

  private:
    double _weight;
};

struct PointsDownwardsPrior_FixedPosition
{
    static const int NUM_RESIDUALS = 1;
    static const int NUM_PARAMETERS_1 = ORIENTATION_PARAMETERS;

    PointsDownwardsPrior_FixedPosition(double weight) : _weight(weight)
    {
    }

    template <typename T> bool operator()(const T *rotation, T *residuals) const
    {
        residuals[0] = pointsDownwardsResidual(rotation, _weight);
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

struct ValuePrior
{
    static const int NUM_RESIDUALS = 1;
    static const int NUM_PARAMETERS_1 = 1;

    ValuePrior(double target, double weight) : _target(target), _weight(weight)
    {
    }

    template <typename T> bool operator()(const T *val, T *residual) const
    {
        residual[0] = T(_weight) * (val[0] - T(_target));
        return true;
    }

  private:
    const double _target;
    const double _weight;
};

struct GPSPositionPrior
{
    static const int NUM_RESIDUALS = 3;
    static const int NUM_PARAMETERS_1 = POSE_PARAMETERS;

    GPSPositionPrior(const Eigen::Vector3d &gps_position, double horizontal_weight, double vertical_weight)
        : _gps_position(gps_position), _weights(horizontal_weight, horizontal_weight, vertical_weight)
    {
    }

    template <typename T> bool operator()(const T *pose, T *residuals) const
    {
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using Vector3TCM = Eigen::Map<const Vector3T>;
        using Vector3TM = Eigen::Map<Vector3T>;

        const Vector3TCM position_em(pose + 4);

        Vector3TM residuals_m(residuals);
        residuals_m = _weights.cast<T>().cwiseProduct(position_em - _gps_position.cast<T>());
        return true;
    }

  private:
    const Eigen::Vector3d _gps_position;
    const Eigen::Vector3d _weights;
};

template <typename T>
Eigen::Matrix<T, 3, 1> robustCentroid(const Eigen::Matrix<T, 3, 1> *points, int n, T outlier_distance)
{
    using Vector3T = Eigen::Matrix<T, 3, 1>;
    constexpr int MAX_STAGES = 3;

    Vector3T centroid = Vector3T::Zero();
    for (int i = 0; i < n; i++)
        centroid += points[i];
    centroid /= T(n);

    for (int stage = 0; stage < MAX_STAGES; stage++)
    {
        T total_w = T(0);
        Vector3T weighted_sum = Vector3T::Zero();
        for (int i = 0; i < n; i++)
        {
            T err = (points[i] - centroid).norm();
            T w = T(1.0) / (err + T(1e-8));
            if (err > outlier_distance)
                w *= outlier_distance / err;
            total_w += w;
            weighted_sum += w * points[i];
        }
        centroid = weighted_sum / total_w;
    }

    return centroid;
}

struct AdjacentTriangleNormalCost
{
    static const int NUM_RESIDUALS = 1;
    static const int NUM_PARAMETERS_1 = 1; // z of edge node A
    static const int NUM_PARAMETERS_2 = 1; // z of edge node B
    static const int NUM_PARAMETERS_3 = 1; // z of opposite node C (triangle 1)
    static const int NUM_PARAMETERS_4 = 1; // z of opposite node D (triangle 2)

    AdjacentTriangleNormalCost(const Eigen::Vector2d &xyA, const Eigen::Vector2d &xyB, const Eigen::Vector2d &xyC,
                               const Eigen::Vector2d &xyD, double weight)
        : _xyA(xyA), _xyB(xyB), _xyC(xyC), _xyD(xyD), _weight(weight)
    {
    }

    template <typename T> bool operator()(const T *zA, const T *zB, const T *zC, const T *zD, T *residuals) const
    {
        using Vector3T = Eigen::Matrix<T, 3, 1>;

        const Vector3T A(T(_xyA.x()), T(_xyA.y()), *zA);
        const Vector3T B(T(_xyB.x()), T(_xyB.y()), *zB);
        const Vector3T C(T(_xyC.x()), T(_xyC.y()), *zC);
        const Vector3T D(T(_xyD.x()), T(_xyD.y()), *zD);

        residuals[0] = T(_weight) * signedDihedralAngle<T>(A, B, C, D);
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  private:
    const Eigen::Vector2d _xyA, _xyB, _xyC, _xyD;
    const double _weight;
};

struct MeshPointHeightCost
{
    static const int NUM_RESIDUALS = 1;
    static const int NUM_PARAMETERS_1 = 1;
    static const int NUM_PARAMETERS_2 = 1;
    static const int NUM_PARAMETERS_3 = 1;

    MeshPointHeightCost(const Eigen::Vector3d &barycentric, double z, double weight)
        : _barycentric(barycentric), _z(z), _weight(weight)
    {
    }

    template <typename T> bool operator()(const T *zA, const T *zB, const T *zC, T *residuals) const
    {
        const T meshHeight = T(_barycentric[0]) * *zA + T(_barycentric[1]) * *zB + T(_barycentric[2]) * *zC;
        residuals[0] = T(_weight) * (T(_z) - meshHeight);
        return true;
    }

  private:
    const Eigen::Vector3d _barycentric;
    const double _z, _weight;
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
    static const int NUM_PARAMETERS_1 = POSE_PARAMETERS;
    static const int NUM_PARAMETERS_2 = POSE_PARAMETERS;

    DecomposedRotationCost(const Eigen::Quaterniond &relative_rotation, const Eigen::Vector3d &relative_translation,
                           int score)
        : _has_relative_translation(relative_translation.squaredNorm() > 1e-9),
          _relative_rotation(relative_rotation.normalized()),
          _relative_translation_direction(relative_translation.normalized()), _weight(std::sqrt(score / 8.))
    {
    }

    template <typename T> bool operator()(const T *pose1, const T *pose2, T *residuals) const
    {
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        return givenTranslation(pose1, pose2,
                                Vector3T(Eigen::Map<const Vector3T>(pose2 + 4) - Eigen::Map<const Vector3T>(pose1 + 4)),
                                residuals);
    }

    template <typename T>
    bool givenTranslation(const T *rotation1, const T *rotation2, const Eigen::Matrix<T, 3, 1> &translation,
                          T *residuals) const
    {
        using QuaterionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using QuaterionTCM = Eigen::Map<const QuaterionT>;

        const QuaterionTCM rotation1_em(rotation1);
        const QuaterionTCM rotation2_em(rotation2);

        T res[3];

        if (_has_relative_translation && translation.squaredNorm() > T(1e-9))
        {
            const Vector3T translation_direction = translation.normalized();

            // angle from camera1 -> camera2
            const Vector3T rotated_translation2_1 = rotation1_em.inverse() * translation_direction;
            res[0] = angleBetweenUnitVectors<T>(rotated_translation2_1, _relative_translation_direction.cast<T>());

            // angle from camera2 -> camera1
            const Vector3T rotated_translation1_2 =
                rotation2_em.inverse() * (_relative_rotation.cast<T>() * -translation_direction);
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
    const bool _has_relative_translation;
    const Eigen::Quaterniond _relative_rotation;
    const Eigen::Vector3d _relative_translation_direction;
    const double _weight;
};

// Homographies give two valid decompositions. Use this to evaluate the cost of these paired decompositions.
// Takes the residual from the decomposition with lower error, or whichever doesn't have NaN/Inf in it
inline std::vector<DecomposedRotationCost> strongDecompositions(const camera_relations &relations)
{
    int max_score = 0;
    for (const auto &pose : relations.relative_poses)
        max_score = std::max(max_score, pose.score);

    std::vector<DecomposedRotationCost> decompositions;
    decompositions.reserve(relations.relative_poses.size());
    for (const auto &pose : relations.relative_poses)
        if (pose.score > 0.25 * max_score)
            decompositions.emplace_back(pose.orientation, pose.position, pose.score);
    return decompositions;
}

template <typename T>
bool lowestDecompositionResidual(const std::vector<DecomposedRotationCost> &decompositions, const T *rotation1,
                                 const T *rotation2, const Eigen::Matrix<T, 3, 1> &translation, T *residuals)
{
    using VectorRT = Eigen::Matrix<T, DecomposedRotationCost::NUM_RESIDUALS, 1>;

    T lowest_res_norm(std::numeric_limits<double>::infinity());
    VectorRT lowest_res;
    lowest_res.fill(T(NAN));
    for (const auto &d : decompositions)
    {
        VectorRT res;
        if (d.givenTranslation(rotation1, rotation2, translation, res.data()) && res.allFinite() &&
            res.squaredNorm() < lowest_res_norm)
        {
            lowest_res_norm = res.squaredNorm();
            lowest_res = res;
        }
    }

    Eigen::Map<VectorRT>{residuals} = lowest_res;
    return ceres::isfinite(lowest_res_norm);
}

struct MultiDecomposedRotationCost
{
    static const int NUM_RESIDUALS = DecomposedRotationCost::NUM_RESIDUALS;
    static const int NUM_PARAMETERS_1 = POSE_PARAMETERS;
    static const int NUM_PARAMETERS_2 = POSE_PARAMETERS;

    MultiDecomposedRotationCost(const camera_relations &relations) : decompose(strongDecompositions(relations))
    {
    }

    template <typename T> bool operator()(const T *pose1, const T *pose2, T *residuals) const
    {
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        const Vector3T translation = Eigen::Map<const Vector3T>(pose2 + 4) - Eigen::Map<const Vector3T>(pose1 + 4);
        return lowestDecompositionResidual(decompose, pose1, pose2, translation, residuals);
    }

    std::vector<DecomposedRotationCost> decompose;
};

struct MultiDecomposedRotationCost_FixedPositions
{
    static const int NUM_RESIDUALS = DecomposedRotationCost::NUM_RESIDUALS;
    static const int NUM_PARAMETERS_1 = ORIENTATION_PARAMETERS;
    static const int NUM_PARAMETERS_2 = ORIENTATION_PARAMETERS;

    MultiDecomposedRotationCost_FixedPositions(const camera_relations &relations,
                                               const Eigen::Vector3d &dest_minus_source)
        : decompose(strongDecompositions(relations)), translation(dest_minus_source)
    {
    }

    template <typename T> bool operator()(const T *rotation1, const T *rotation2, T *residuals) const
    {
        return lowestDecompositionResidual(decompose, rotation1, rotation2, translation.cast<T>().eval(), residuals);
    }

    std::vector<DecomposedRotationCost> decompose;
    Eigen::Vector3d translation;
};

template <typename Cost, int N> struct PoseBlocks
{
    template <typename T> bool operator()(const T *p0, const T *p1, T *res) const
    {
        static_assert(N == 2);
        const T *poses[]{p0, p1};
        return static_cast<const Cost &>(*this).eval(poses, res);
    }

    template <typename T> bool operator()(const T *p0, const T *p1, const T *p2, T *res) const
    {
        static_assert(N == 3);
        const T *poses[]{p0, p1, p2};
        return static_cast<const Cost &>(*this).eval(poses, res);
    }

    template <typename T> bool operator()(const T *p0, const T *p1, const T *p2, const T *p3, T *res) const
    {
        static_assert(N == 4);
        const T *poses[]{p0, p1, p2, p3};
        return static_cast<const Cost &>(*this).eval(poses, res);
    }

    template <typename T> bool operator()(const T *p0, const T *p1, const T *p2, const T *p3, const T *p4, T *res) const
    {
        static_assert(N == 5);
        const T *poses[]{p0, p1, p2, p3, p4};
        return static_cast<const Cost &>(*this).eval(poses, res);
    }
};

template <int N> struct TriangulatedReprojectionResiduals
{
    static_assert(N >= 2 && N <= 5, "N must be between 2 and 5");
    static const int NUM_RESIDUALS = N * 3;

    TriangulatedReprojectionResiduals(const std::array<Eigen::Vector3d, N> &camera_rays,
                                      const std::array<double, N> &inverse_sigmas)
        : camera_ray(camera_rays), inverse_sigma(inverse_sigmas)
    {
    }

    template <typename T>
    bool computeResiduals(const T *const *rotations, const Eigen::Matrix<T, 3, 1> *positions, T *residuals) const
    {
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using Matrix3T = Eigen::Matrix<T, 3, 3>;
        using QuaternionTCM = Eigen::Map<const Eigen::Quaternion<T>>;
        using Vector3TM = Eigen::Map<Vector3T>;

        std::array<Vector3T, N> dirs;
        for (int i = 0; i < N; i++)
            dirs[i] = (QuaternionTCM(rotations[i]) * camera_ray[i].template cast<T>()).normalized();

        const auto triangulate = [&](const std::array<T, N> &weights, Vector3T &point) {
            Matrix3T normal_matrix = Matrix3T::Zero();
            Vector3T normal_rhs = Vector3T::Zero();
            for (int i = 0; i < N; i++)
            {
                const Matrix3T perpendicular = weights[i] * (Matrix3T::Identity() - dirs[i] * dirs[i].transpose());
                normal_matrix += perpendicular;
                normal_rhs += perpendicular * positions[i];
            }
            if (!(normal_matrix.determinant() > T(1e-12)))
                return false;
            point = normal_matrix.inverse() * normal_rhs;
            return true;
        };

        std::array<T, N> weights;
        weights.fill(T(1));
        Vector3T point;
        if (!triangulate(weights, point))
            return false;

        if constexpr (N > 2)
        {
            const T cos_5_degrees(0.996);
            bool any_outlier = false;
            for (int i = 0; i < N; i++)
                any_outlier |= dirs[i].dot((point - positions[i]).normalized()) < cos_5_degrees;
            if (any_outlier)
            {
                for (int i = 0; i < N; i++)
                {
                    std::array<T, N> all_but_i;
                    all_but_i.fill(T(1));
                    all_but_i[i] = T(0);
                    Vector3T point_seen_by_others;
                    weights[i] = triangulate(all_but_i, point_seen_by_others)
                                     ? (T(1) + dirs[i].dot((point_seen_by_others - positions[i]).normalized())) * T(0.5)
                                     : T(1);
                }
                Vector3T weighted_point;
                if (triangulate(weights, weighted_point))
                    point = weighted_point;
            }
        }

        for (int i = 0; i < N; i++)
        {
            const Vector3T p_cam = QuaternionTCM(rotations[i]).inverse() * (point - positions[i]);
            const Vector3T chord = p_cam.normalized() - camera_ray[i].template cast<T>().normalized();
            Vector3TM(residuals + i * 3) = chordScaledToAngle(chord) * T(inverse_sigma[i]);
        }
        return true;
    }

    template <typename T> static Eigen::Matrix<T, 3, 1> chordScaledToAngle(const Eigen::Matrix<T, 3, 1> &chord)
    {
        using std::asin;
        using std::sqrt;
        const T half_chord_sq = chord.squaredNorm() * T(0.25);
        const T half_chord = sqrt(ceres::fmin(half_chord_sq, T(1 - 1e-9)));
        const T scale = half_chord_sq < T(1e-8) ? T(1) + half_chord_sq / T(6) : asin(half_chord) / half_chord;
        return chord * scale;
    }

    const std::array<Eigen::Vector3d, N> camera_ray;
    const std::array<double, N> inverse_sigma;
};

template <int N> struct TriangulatedReprojectionCost : PoseBlocks<TriangulatedReprojectionCost<N>, N>
{
    static const int NUM_RESIDUALS = TriangulatedReprojectionResiduals<N>::NUM_RESIDUALS;

    TriangulatedReprojectionCost(const std::array<Eigen::Vector3d, N> &camera_rays,
                                 const std::array<double, N> &inverse_sigmas = unitSigmas<N>())
        : _impl(camera_rays, inverse_sigmas)
    {
    }

    template <typename T> bool eval(const T *const *poses, T *residuals) const
    {
        const auto positions = posePositions<N>(poses);
        return _impl.computeResiduals(poses, positions.data(), residuals);
    }

  private:
    TriangulatedReprojectionResiduals<N> _impl;
};

template <int N>
struct TriangulatedReprojectionCost_FixedPositions : PoseBlocks<TriangulatedReprojectionCost_FixedPositions<N>, N>
{
    static const int NUM_RESIDUALS = TriangulatedReprojectionResiduals<N>::NUM_RESIDUALS;

    TriangulatedReprojectionCost_FixedPositions(const std::array<Eigen::Vector3d, N> &camera_rays,
                                                const std::array<Eigen::Vector3d, N> &camera_positions,
                                                const std::array<double, N> &inverse_sigmas = unitSigmas<N>())
        : _impl(camera_rays, inverse_sigmas), _positions(camera_positions)
    {
    }

    template <typename T> bool eval(const T *const *rotations, T *residuals) const
    {
        const auto positions = castPositions<T>(_positions);
        return _impl.computeResiduals(rotations, positions.data(), residuals);
    }

  private:
    TriangulatedReprojectionResiduals<N> _impl;
    std::array<Eigen::Vector3d, N> _positions;
};

template <typename T>
void pixelResidual(const Eigen::Matrix<T, 3, 1> &ray, const DifferentiableCameraModel<T> &model,
                   const Eigen::Vector2d &pixel, T *residuals)
{
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals_m(residuals);
    residuals_m = image_from_3d<T>(ray, model) - pixel.cast<T>();
    const T behind = T(MIN_PROJECTION_Z) - ray.z();
    if (behind > T(0))
        residuals_m.array() += model.focal_length_pixels * behind;
}

template <typename T> Eigen::Matrix<T, 3, 1> cameraRay(const T *pose, const T *point)
{
    const Eigen::Map<const Eigen::Quaternion<T>> rotation(pose);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> location(pose + 4);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> point_m(point);
    return rotation.inverse() * (point_m - location);
}

struct PixelErrorCost_Orientation
{
    static const int NUM_RESIDUALS = 2;
    static const int NUM_PARAMETERS_1 = POSE_PARAMETERS;
    static const int NUM_PARAMETERS_2 = 3;

    PixelErrorCost_Orientation(const CameraModel &camera_model, const Eigen::Vector2d &camera_pixel)
        : model(camera_model), pixel(camera_pixel)
    {
    }

    template <typename T> bool operator()(const T *pose, const T *point, T *residuals) const
    {
        pixelResidual<T>(cameraRay(pose, point), model.cast<T>(), pixel, residuals);
        return true;
    }

  private:
    const CameraModel &model;
    const Eigen::Vector2d pixel; // unique to this measurement, keep a local copy to avoid cache thrashing
};

struct PixelErrorCost_OrientationFocal
{
    static const int NUM_RESIDUALS = 2;
    static const int NUM_PARAMETERS_1 = POSE_PARAMETERS;
    static const int NUM_PARAMETERS_2 = 3;
    static const int NUM_PARAMETERS_3 = 1;
    static const int NUM_PARAMETERS_4 = 2;

    PixelErrorCost_OrientationFocal(const CameraModel &camera_model, const Eigen::Vector2d &camera_pixel)
        : model(camera_model), pixel(camera_pixel)
    {
    }

    template <typename T>
    bool operator()(const T *pose, const T *point, const T *focal, const T *principal, T *residuals) const
    {
        DifferentiableCameraModel<T> model_t = model.cast<T>();
        model_t.focal_length_pixels = *focal;
        model_t.principle_point = Eigen::Map<const Eigen::Matrix<T, 2, 1>>(principal);
        pixelResidual<T>(cameraRay(pose, point), model_t, pixel, residuals);
        return true;
    }

  private:
    const CameraModel &model;
    const Eigen::Vector2d pixel; // unique to this measurement, keep a local copy to avoid cache thrashing
};

struct PixelErrorCost_OrientationFocalRadial
{
    static const int NUM_RESIDUALS = 2;
    static const int NUM_PARAMETERS_1 = POSE_PARAMETERS;
    static const int NUM_PARAMETERS_2 = 3;
    static const int NUM_PARAMETERS_3 = 1;
    static const int NUM_PARAMETERS_4 = 2;
    static const int NUM_PARAMETERS_5 = 3;

    PixelErrorCost_OrientationFocalRadial(const CameraModel &camera_model, const Eigen::Vector2d &camera_pixel)
        : model(camera_model), pixel(camera_pixel)
    {
    }

    template <typename T>
    bool operator()(const T *pose, const T *point, const T *focal, const T *principal, const T *radial,
                    T *residuals) const
    {
        DifferentiableCameraModel<T> model_t = model.cast<T>();
        model_t.focal_length_pixels = *focal;
        model_t.principle_point = Eigen::Map<const Eigen::Matrix<T, 2, 1>>(principal);
        model_t.radial_distortion = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(radial);
        pixelResidual<T>(cameraRay(pose, point), model_t, pixel, residuals);
        return true;
    }

  private:
    const CameraModel &model;
    const Eigen::Vector2d pixel; // unique to this measurement, keep a local copy to avoid cache thrashing
};

struct PixelErrorCost_OrientationFocalRadialTangential
{
    static const int NUM_RESIDUALS = 2;
    static const int NUM_PARAMETERS_1 = POSE_PARAMETERS;
    static const int NUM_PARAMETERS_2 = 3;
    static const int NUM_PARAMETERS_3 = 1;
    static const int NUM_PARAMETERS_4 = 2;
    static const int NUM_PARAMETERS_5 = 3;
    static const int NUM_PARAMETERS_6 = 2;

    PixelErrorCost_OrientationFocalRadialTangential(const CameraModel &camera_model,
                                                    const Eigen::Vector2d &camera_pixel)
        : model(camera_model), pixel(camera_pixel)
    {
    }

    template <typename T>
    bool operator()(const T *pose, const T *point, const T *focal, const T *principal, const T *radial,
                    const T *tangential, T *residuals) const
    {
        DifferentiableCameraModel<T> model_t = model.cast<T>();
        model_t.focal_length_pixels = *focal;
        model_t.principle_point = Eigen::Map<const Eigen::Matrix<T, 2, 1>>(principal);
        model_t.radial_distortion = Eigen::Map<const Eigen::Matrix<T, 3, 1>>(radial);
        model_t.tangential_distortion = Eigen::Map<const Eigen::Matrix<T, 2, 1>>(tangential);
        pixelResidual<T>(cameraRay(pose, point), model_t, pixel, residuals);
        return true;
    }

  private:
    const CameraModel &model;
    const Eigen::Vector2d pixel; // unique to this measurement, keep a local copy to avoid cache thrashing
};

template <int N> struct MultiRayPlaneIntersectionAngleCost_FocalRadial
{
    static_assert(N >= 2 && N <= 5, "N must be between 2 and 5");
    static const int NUM_RESIDUALS = N * 2;

    MultiRayPlaneIntersectionAngleCost_FocalRadial(const std::array<Eigen::Vector2d, N> &camera_pixels,
                                                   const std::array<Eigen::Vector2d, 3> &plane_points,
                                                   const InverseDifferentiableCameraModel<double> &model)
        : camera_pixel(camera_pixels), plane_point(plane_points), sharedModel(model)
    {
    }

    template <typename T>
    bool computeResiduals(const T *const *rotations, const Eigen::Matrix<T, 3, 1> *positions, const T *z0, const T *z1,
                          const T *z2, const T *focal, const T *principal, const T *radial, T *residuals) const
    {
        using QuaternionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using QuaternionTCM = Eigen::Map<const QuaternionT>;
        using Vector2TM = Eigen::Map<Eigen::Matrix<T, 2, 1>>;
        using Vector3TCM = Eigen::Map<const Vector3T>;
        using Vector2TCM = Eigen::Map<const Eigen::Matrix<T, 2, 1>>;

        InverseDifferentiableCameraModel<T> model = sharedModel.template cast<T>();
        model.focal_length_pixels = T(*focal);
        model.principle_point = Vector2TCM(principal);
        model.radial_distortion = Vector3TCM(radial);

        const T plane_z[3]{*z0, *z1, *z2};
        plane_3_corners<T> plane3;
        for (int i = 0; i < 3; i++)
            plane3.corner[i] << T(plane_point[i].x()), T(plane_point[i].y()), plane_z[i];

        plane_norm_offset<T> pno = cornerPlane2normOffsetPlane(plane3);

        Vector3T intersection[N];
        Vector3T camera_ray[N];
        bool all_valid = true;
        T avg_dist = T(0);
        for (int i = 0; i < N; i++)
        {
            const QuaternionTCM rot(rotations[i]);
            camera_ray[i] = image_to_3d<T>(camera_pixel[i].template cast<T>(), model);
            ray<T> r;
            r.dir = rot * camera_ray[i];
            r.offset = positions[i];
            all_valid &= rayPlaneIntersection(r, pno, intersection[i]);
            avg_dist += (intersection[i] - r.offset).norm();
        }
        avg_dist /= T(N);

        T outlier_distance = avg_dist * T(0.01);
        Vector3T centroid = robustCentroid(intersection, N, outlier_distance);

        const T inverse_sigma = *focal / RAY_PIXEL_SIGMA;
        for (int i = 0; i < N; i++)
        {
            const QuaternionTCM rot(rotations[i]);
            const Vector3T p_cam = rot.inverse() * (centroid - positions[i]);
            const Vector3T &ray = camera_ray[i];
            const T min_depth = T(0.5) * p_cam.norm() * ray.z() / ray.norm();
            const T depth = p_cam.z() > min_depth ? p_cam.z() : min_depth;
            Vector2TM(residuals + i * 2) =
                (p_cam.template head<2>() / depth - ray.template head<2>() / ray.z()) * inverse_sigma;
        }

        return all_valid;
    }

    const std::array<Eigen::Vector2d, N> camera_pixel;
    const std::array<Eigen::Vector2d, 3> plane_point;
    const InverseDifferentiableCameraModel<double> sharedModel;
};

struct PlaneIntersectionAngleCost_OrientationFocalRadial_SharedModel
{
    static const int NUM_RESIDUALS = MultiRayPlaneIntersectionAngleCost_FocalRadial<2>::NUM_RESIDUALS;
    static const int NUM_PARAMETERS_1 = POSE_PARAMETERS; // pose 0
    static const int NUM_PARAMETERS_2 = POSE_PARAMETERS; // pose 1
    static const int NUM_PARAMETERS_3 = 1;               // z 0
    static const int NUM_PARAMETERS_4 = 1;               // z 1
    static const int NUM_PARAMETERS_5 = 1;               // z 2
    static const int NUM_PARAMETERS_6 = 1;               // focal
    static const int NUM_PARAMETERS_7 = 2;               // principal
    static const int NUM_PARAMETERS_8 = 3;               // radial

    PlaneIntersectionAngleCost_OrientationFocalRadial_SharedModel(
        const Eigen::Vector2d &camera_pixel1, const Eigen::Vector2d &camera_pixel2, const Eigen::Vector2d &plane_point1,
        const Eigen::Vector2d &plane_point2, const Eigen::Vector2d &plane_point3,
        const InverseDifferentiableCameraModel<double> &sharedModel)
        : _impl({camera_pixel1, camera_pixel2}, {{plane_point1, plane_point2, plane_point3}}, sharedModel)
    {
    }

    template <typename T>
    bool operator()(const T *pose0, const T *pose1, const T *z0, const T *z1, const T *z2, const T *focal,
                    const T *principal, const T *radial, T *residuals) const
    {
        const T *poses[2]{pose0, pose1};
        const auto positions = posePositions<2>(poses);
        return _impl.computeResiduals(poses, positions.data(), z0, z1, z2, focal, principal, radial, residuals);
    }

  private:
    MultiRayPlaneIntersectionAngleCost_FocalRadial<2> _impl;
};

struct PlaneIntersectionAngleCost_OrientationFocalRadial_SharedModel_FixedPositions
{
    static const int NUM_RESIDUALS = MultiRayPlaneIntersectionAngleCost_FocalRadial<2>::NUM_RESIDUALS;
    static const int NUM_PARAMETERS_1 = ORIENTATION_PARAMETERS;
    static const int NUM_PARAMETERS_2 = ORIENTATION_PARAMETERS;
    static const int NUM_PARAMETERS_3 = 1;
    static const int NUM_PARAMETERS_4 = 1;
    static const int NUM_PARAMETERS_5 = 1;
    static const int NUM_PARAMETERS_6 = 1;
    static const int NUM_PARAMETERS_7 = 2;
    static const int NUM_PARAMETERS_8 = 3;

    PlaneIntersectionAngleCost_OrientationFocalRadial_SharedModel_FixedPositions(
        const Eigen::Vector2d &camera_pixel1, const Eigen::Vector2d &camera_pixel2, const Eigen::Vector2d &plane_point1,
        const Eigen::Vector2d &plane_point2, const Eigen::Vector2d &plane_point3,
        const InverseDifferentiableCameraModel<double> &sharedModel,
        const std::array<Eigen::Vector3d, 2> &camera_positions)
        : _impl({camera_pixel1, camera_pixel2}, {{plane_point1, plane_point2, plane_point3}}, sharedModel),
          _positions(camera_positions)
    {
    }

    template <typename T>
    bool operator()(const T *rotation0, const T *rotation1, const T *z0, const T *z1, const T *z2, const T *focal,
                    const T *principal, const T *radial, T *residuals) const
    {
        const T *rotations[2]{rotation0, rotation1};
        const auto positions = castPositions<T>(_positions);
        return _impl.computeResiduals(rotations, positions.data(), z0, z1, z2, focal, principal, radial, residuals);
    }

  private:
    MultiRayPlaneIntersectionAngleCost_FocalRadial<2> _impl;
    std::array<Eigen::Vector3d, 2> _positions;
};

template <int N> struct MultiRayPlaneIntersectionAngleCost
{
    static_assert(N >= 2 && N <= 5, "N must be between 2 and 5");
    static const int NUM_RESIDUALS = N * 3;

    MultiRayPlaneIntersectionAngleCost(const std::array<Eigen::Vector3d, N> &camera_rays,
                                       const std::array<Eigen::Vector2d, 3> &plane_points,
                                       const std::array<double, N> &inverse_sigmas = unitSigmas<N>())
        : camera_ray(camera_rays), plane_point(plane_points), inverse_sigma(inverse_sigmas)
    {
    }

    template <typename T>
    bool computeResiduals(const T *const *rotations, const Eigen::Matrix<T, 3, 1> *positions, const T *z0, const T *z1,
                          const T *z2, T *residuals) const
    {
        using QuaternionT = Eigen::Quaternion<T>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;
        using QuaternionTCM = Eigen::Map<const QuaternionT>;
        using Vector3TM = Eigen::Map<Vector3T>;

        const T plane_z[3]{*z0, *z1, *z2};
        plane_3_corners<T> plane3;
        for (int i = 0; i < 3; i++)
            plane3.corner[i] << T(plane_point[i].x()), T(plane_point[i].y()), plane_z[i];

        plane_norm_offset<T> pno = cornerPlane2normOffsetPlane(plane3);

        Vector3T intersection[N];
        bool all_valid = true;
        T avg_dist = T(0);
        for (int i = 0; i < N; i++)
        {
            const QuaternionTCM rot(rotations[i]);
            ray<T> r;
            r.dir = rot * camera_ray[i].template cast<T>();
            r.offset = positions[i];
            all_valid &= rayPlaneIntersection(r, pno, intersection[i]);
            avg_dist += (intersection[i] - r.offset).norm();
        }
        avg_dist /= T(N);

        T outlier_distance = avg_dist * T(0.01);
        Vector3T centroid = robustCentroid(intersection, N, outlier_distance);

        for (int i = 0; i < N; i++)
        {
            Vector3TM(residuals + i * 3) = (intersection[i] - centroid) * (inverse_sigma[i] / avg_dist);
        }

        return all_valid;
    }

    const std::array<Eigen::Vector3d, N> camera_ray;
    const std::array<Eigen::Vector2d, 3> plane_point;
    const std::array<double, N> inverse_sigma;
};

struct PlaneIntersectionAngleCost
{
    static const int NUM_RESIDUALS = MultiRayPlaneIntersectionAngleCost<2>::NUM_RESIDUALS;
    static const int NUM_PARAMETERS_1 = POSE_PARAMETERS;
    static const int NUM_PARAMETERS_2 = POSE_PARAMETERS;
    static const int NUM_PARAMETERS_3 = 1;
    static const int NUM_PARAMETERS_4 = 1;
    static const int NUM_PARAMETERS_5 = 1;

    PlaneIntersectionAngleCost(const Eigen::Vector3d &camera_ray1, const Eigen::Vector3d &camera_ray2,
                               const Eigen::Vector2d &plane_point1, const Eigen::Vector2d &plane_point2,
                               const Eigen::Vector2d &plane_point3,
                               const std::array<double, 2> &inverse_sigmas = unitSigmas<2>())
        : _impl({camera_ray1, camera_ray2}, {{plane_point1, plane_point2, plane_point3}}, inverse_sigmas)
    {
    }

    template <typename T>
    bool operator()(const T *pose0, const T *pose1, const T *z0, const T *z1, const T *z2, T *residuals) const
    {
        const T *poses[2]{pose0, pose1};
        const auto positions = posePositions<2>(poses);
        return _impl.computeResiduals(poses, positions.data(), z0, z1, z2, residuals);
    }

  private:
    MultiRayPlaneIntersectionAngleCost<2> _impl;
};

struct PlaneIntersectionAngleCost_FixedPositions
{
    static const int NUM_RESIDUALS = MultiRayPlaneIntersectionAngleCost<2>::NUM_RESIDUALS;
    static const int NUM_PARAMETERS_1 = ORIENTATION_PARAMETERS;
    static const int NUM_PARAMETERS_2 = ORIENTATION_PARAMETERS;
    static const int NUM_PARAMETERS_3 = 1;
    static const int NUM_PARAMETERS_4 = 1;
    static const int NUM_PARAMETERS_5 = 1;

    PlaneIntersectionAngleCost_FixedPositions(const Eigen::Vector3d &camera_ray1, const Eigen::Vector3d &camera_ray2,
                                              const Eigen::Vector2d &plane_point1, const Eigen::Vector2d &plane_point2,
                                              const Eigen::Vector2d &plane_point3,
                                              const std::array<Eigen::Vector3d, 2> &camera_positions,
                                              const std::array<double, 2> &inverse_sigmas = unitSigmas<2>())
        : _impl({camera_ray1, camera_ray2}, {{plane_point1, plane_point2, plane_point3}}, inverse_sigmas),
          _positions(camera_positions)
    {
    }

    template <typename T>
    bool operator()(const T *rotation0, const T *rotation1, const T *z0, const T *z1, const T *z2, T *residuals) const
    {
        const T *rotations[2]{rotation0, rotation1};
        const auto positions = castPositions<T>(_positions);
        return _impl.computeResiduals(rotations, positions.data(), z0, z1, z2, residuals);
    }

  private:
    MultiRayPlaneIntersectionAngleCost<2> _impl;
    std::array<Eigen::Vector3d, 2> _positions;
};

template <typename Cost, int N> struct ZHeightsThenPoseBlocks
{
    template <typename T>
    bool operator()(const T *z0, const T *z1, const T *z2, const T *r0, const T *r1, const T *r2, T *res) const
    {
        static_assert(N == 3);
        const T *rots[]{r0, r1, r2};
        return static_cast<const Cost &>(*this).eval(z0, z1, z2, rots, res);
    }

    template <typename T>
    bool operator()(const T *z0, const T *z1, const T *z2, const T *r0, const T *r1, const T *r2, const T *r3,
                    T *res) const
    {
        static_assert(N == 4);
        const T *rots[]{r0, r1, r2, r3};
        return static_cast<const Cost &>(*this).eval(z0, z1, z2, rots, res);
    }

    template <typename T>
    bool operator()(const T *z0, const T *z1, const T *z2, const T *r0, const T *r1, const T *r2, const T *r3,
                    const T *r4, T *res) const
    {
        static_assert(N == 5);
        const T *rots[]{r0, r1, r2, r3, r4};
        return static_cast<const Cost &>(*this).eval(z0, z1, z2, rots, res);
    }
};

template <int N> struct PlaneIntersectionAngleCost_NRay : ZHeightsThenPoseBlocks<PlaneIntersectionAngleCost_NRay<N>, N>
{
    static_assert(N >= 3 && N <= 5, "N must be between 3 and 5");
    static const int NUM_RESIDUALS = N * 3;

    PlaneIntersectionAngleCost_NRay(const std::array<Eigen::Vector3d, N> &rays,
                                    const std::array<Eigen::Vector2d, 3> &plane_points,
                                    const std::array<double, N> &inverse_sigmas = unitSigmas<N>())
        : _impl(rays, plane_points, inverse_sigmas)
    {
    }

    template <typename T> bool eval(const T *z0, const T *z1, const T *z2, const T *const *poses, T *res) const
    {
        const auto positions = posePositions<N>(poses);
        return _impl.computeResiduals(poses, positions.data(), z0, z1, z2, res);
    }

  private:
    MultiRayPlaneIntersectionAngleCost<N> _impl;
};

template <int N>
struct PlaneIntersectionAngleCost_NRay_FixedPositions
    : ZHeightsThenPoseBlocks<PlaneIntersectionAngleCost_NRay_FixedPositions<N>, N>
{
    static_assert(N >= 3 && N <= 5, "N must be between 3 and 5");
    static const int NUM_RESIDUALS = N * 3;

    PlaneIntersectionAngleCost_NRay_FixedPositions(const std::array<Eigen::Vector3d, N> &rays,
                                                   const std::array<Eigen::Vector2d, 3> &plane_points,
                                                   const std::array<Eigen::Vector3d, N> &camera_positions,
                                                   const std::array<double, N> &inverse_sigmas = unitSigmas<N>())
        : _impl(rays, plane_points, inverse_sigmas), _positions(camera_positions)
    {
    }

    template <typename T> bool eval(const T *z0, const T *z1, const T *z2, const T *const *rotations, T *res) const
    {
        const auto positions = castPositions<T>(_positions);
        return _impl.computeResiduals(rotations, positions.data(), z0, z1, z2, res);
    }

  private:
    MultiRayPlaneIntersectionAngleCost<N> _impl;
    std::array<Eigen::Vector3d, N> _positions;
};

template <typename Cost, int N> struct ZHeightsIntrinsicsThenPoseBlocks
{
    template <typename T>
    bool operator()(const T *z0, const T *z1, const T *z2, const T *focal, const T *principal, const T *radial,
                    const T *r0, const T *r1, const T *r2, T *res) const
    {
        static_assert(N == 3);
        const T *rots[]{r0, r1, r2};
        return static_cast<const Cost &>(*this).eval(z0, z1, z2, focal, principal, radial, rots, res);
    }

    template <typename T>
    bool operator()(const T *z0, const T *z1, const T *z2, const T *focal, const T *principal, const T *radial,
                    const T *r0, const T *r1, const T *r2, const T *r3, T *res) const
    {
        static_assert(N == 4);
        const T *rots[]{r0, r1, r2, r3};
        return static_cast<const Cost &>(*this).eval(z0, z1, z2, focal, principal, radial, rots, res);
    }

    template <typename T>
    bool operator()(const T *z0, const T *z1, const T *z2, const T *focal, const T *principal, const T *radial,
                    const T *r0, const T *r1, const T *r2, const T *r3, const T *r4, T *res) const
    {
        static_assert(N == 5);
        const T *rots[]{r0, r1, r2, r3, r4};
        return static_cast<const Cost &>(*this).eval(z0, z1, z2, focal, principal, radial, rots, res);
    }
};

template <int N>
struct PlaneIntersectionAngleCost_NRay_FocalRadial
    : ZHeightsIntrinsicsThenPoseBlocks<PlaneIntersectionAngleCost_NRay_FocalRadial<N>, N>
{
    static_assert(N >= 3 && N <= 5, "N must be between 3 and 5");
    static const int NUM_RESIDUALS = MultiRayPlaneIntersectionAngleCost_FocalRadial<N>::NUM_RESIDUALS;

    PlaneIntersectionAngleCost_NRay_FocalRadial(const std::array<Eigen::Vector2d, N> &pixels,
                                                const std::array<Eigen::Vector2d, 3> &plane_points,
                                                const InverseDifferentiableCameraModel<double> &model)
        : _impl(pixels, plane_points, model)
    {
    }

    template <typename T>
    bool eval(const T *z0, const T *z1, const T *z2, const T *focal, const T *principal, const T *radial,
              const T *const *poses, T *res) const
    {
        const auto positions = posePositions<N>(poses);
        return _impl.computeResiduals(poses, positions.data(), z0, z1, z2, focal, principal, radial, res);
    }

  private:
    MultiRayPlaneIntersectionAngleCost_FocalRadial<N> _impl;
};

template <int N>
struct PlaneIntersectionAngleCost_NRay_FocalRadial_FixedPositions
    : ZHeightsIntrinsicsThenPoseBlocks<PlaneIntersectionAngleCost_NRay_FocalRadial_FixedPositions<N>, N>
{
    static_assert(N >= 3 && N <= 5, "N must be between 3 and 5");
    static const int NUM_RESIDUALS = MultiRayPlaneIntersectionAngleCost_FocalRadial<N>::NUM_RESIDUALS;

    PlaneIntersectionAngleCost_NRay_FocalRadial_FixedPositions(const std::array<Eigen::Vector2d, N> &pixels,
                                                               const std::array<Eigen::Vector2d, 3> &plane_points,
                                                               const InverseDifferentiableCameraModel<double> &model,
                                                               const std::array<Eigen::Vector3d, N> &camera_positions)
        : _impl(pixels, plane_points, model), _positions(camera_positions)
    {
    }

    template <typename T>
    bool eval(const T *z0, const T *z1, const T *z2, const T *focal, const T *principal, const T *radial,
              const T *const *rotations, T *res) const
    {
        const auto positions = castPositions<T>(_positions);
        return _impl.computeResiduals(rotations, positions.data(), z0, z1, z2, focal, principal, radial, res);
    }

  private:
    MultiRayPlaneIntersectionAngleCost_FocalRadial<N> _impl;
    std::array<Eigen::Vector3d, N> _positions;
};

} // namespace opencalibration
