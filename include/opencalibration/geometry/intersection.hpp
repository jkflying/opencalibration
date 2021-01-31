#pragma once

#include <eigen3/Eigen/Core>

namespace opencalibration
{

/**
 * Find midpoint between where the rays pass closest to each other. Return NaN if ambiguous.
 * 4th dimension is used to indicate the squared distance between the rays.
 *
 */
Eigen::Vector4d rayIntersection(const Eigen::Vector3d &origin1, const Eigen::Vector3d &normal1,
                                const Eigen::Vector3d &origin2, const Eigen::Vector3d &normal2);
} // namespace opencalibration
