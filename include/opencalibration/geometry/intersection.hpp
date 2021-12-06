#pragma once

#include <opencalibration/types/camera_model.hpp>
#include <opencalibration/types/plane.hpp>
#include <opencalibration/types/ray.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

namespace opencalibration
{

/**
 * Find midpoint between where the rays pass closest to each other. Return NaN if ambiguous.
 * 4th dimension is used to indicate the squared distance between the rays.
 *
 */
std::pair<Eigen::Vector3d, double> rayIntersection(const ray_d &r1, const ray_d &r2);
std::pair<Eigen::Vector3d, double> rayIntersection(const std::vector<ray_d> &rays);

std::pair<Eigen::Vector3d, double> rayIntersection(const CameraModel &model1, CameraModel &model2,
                                                   const Eigen::Vector3d &pos1, const Eigen::Vector3d &pos2,
                                                   const Eigen::Quaterniond &rot1, const Eigen::Quaterniond &rot2,
                                                   const Eigen::Vector2d &px1, const Eigen::Vector2d &px2);

template <typename T> plane_norm_offset<T> cornerPlane2normOffsetPlane(const plane_3_corners<T> &p3)
{
    plane_norm_offset<T> out;
    out.offset = p3.corner[0];
    out.norm = (p3.corner[0] - p3.corner[1]).cross(p3.corner[0] - p3.corner[2]).normalized();
    return out;
}

template <typename T>
bool rayPlaneIntersection(const ray<T> &r, const plane_norm_offset<T> &p,
                          typename Eigen::Matrix<T, 3, 1> &intersectionPoint)
{
    const T denom = p.norm.dot(r.dir);
    if (abs(denom) < T(1e-9))
    {
        intersectionPoint.fill(T(NAN));
        return false;
    }
    const T t = (p.norm.dot(p.offset) - r.offset.dot(p.norm)) / denom;
    intersectionPoint = r.offset + t * r.dir;
    return true;
}

template <typename T, typename R = Eigen::Matrix<T, 3, 1>>
bool onSameSideOfEdge(const R &vertex0, const R &vertex1, const R &reference, const R &test)
{
    const auto edgeDir = vertex1 - vertex0;
    const auto rawDir = reference - vertex0;
    const auto perpDir = rawDir - edgeDir * (rawDir.dot(edgeDir) / edgeDir.squaredNorm()); // fast but bad numerically
    const auto testDir = test - vertex0;
    const T result = testDir.dot(perpDir);
    return result >= 0;
}

template <typename T>
bool pointInsideTriangle(const typename Eigen::Matrix<T, 3, 1> &point, const plane_3_corners<T> &triangle)
{
    bool inside = true;

    for (size_t i = 0; i < 3; i++)
    {
        inside &=
            onSameSideOfEdge<T>(triangle.corner[i], triangle.corner[(i + 1) % 3], triangle.corner[(i + 2) % 3], point);
    }

    return inside;
}

template <typename T>
bool rayTriangleIntersection(const ray<T> &ray, const plane_3_corners<T> &triangle,
                             typename Eigen::Matrix<T, 3, 1> &intersectionPoint)
{
    const plane_norm_offset<T> plane = cornerPlane2normOffsetPlane(triangle);
    const bool intersects = rayPlaneIntersection(ray, plane, intersectionPoint);
    return intersects && pointInsideTriangle(intersectionPoint, triangle);
}
} // namespace opencalibration
