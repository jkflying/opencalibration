#pragma once

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

template <typename T> plane_norm_offset<T> cornerPlane2normOffsetPlane(const plane_3_corners<T> &p3)
{
    plane_norm_offset<T> out;
    out.offset = p3.corner[0];
    out.norm = (p3.corner[0] - p3.corner[1]).cross(p3.corner[0] - p3.corner[2]).normalized();
    return out;
}

template <typename T>
typename Eigen::Matrix<T, 3, 1> rayPlaneIntersection(const ray<T> &r, const plane_norm_offset<T> &p)
{
    T t = (p.norm.dot(p.offset) - r.offset.dot(p.norm)) / p.norm.dot(r.dir);
    return r.offset + t * r.dir;
}
} // namespace opencalibration
