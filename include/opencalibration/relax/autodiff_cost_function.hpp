#pragma once

#include <opencalibration/types/camera_model.hpp>
#include <opencalibration/types/camera_relations.hpp>

#include <ceres/cost_function.h>
#include <eigen3/Eigen/Core>

namespace opencalibration
{
ceres::CostFunction *newAutoDiffMultiDecomposedRotationCost(const camera_relations &relations,
                                                            const Eigen::Vector3d *translation1,
                                                            const Eigen::Vector3d *translation2);

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost(
    const Eigen::Vector3d &camera_loc1, const Eigen::Vector3d &camera_loc2, const Eigen::Vector3d &camera_ray1,
    const Eigen::Vector3d &camera_ray2, const Eigen::Vector2d &plane_point1, const Eigen::Vector2d &plane_point2,
    const Eigen::Vector2d &plane_point3);

ceres::CostFunction *newAutoDiffPixelErrorCost_Orientation(const Eigen::Vector3d &camera_loc,
                                                           const CameraModel &camera_model,
                                                           const Eigen::Vector2d &camera_pixel);

ceres::CostFunction *newAutoDiffPixelErrorCost_OrientationFocal(const Eigen::Vector3d &camera_loc,
                                                                const CameraModel &camera_model,
                                                                const Eigen::Vector2d &camera_pixel);

ceres::CostFunction *newAutoDiffPixelErrorCost_OrientationFocalRadial(const Eigen::Vector3d &camera_loc,
                                                                      const CameraModel &camera_model,
                                                                      const Eigen::Vector2d &camera_pixel);

ceres::CostFunction *newAutoDiffPixelErrorCost_OrientationFocalRadialTangential(const Eigen::Vector3d &camera_loc,
                                                                                const CameraModel &camera_model,
                                                                                const Eigen::Vector2d &camera_pixel);
ceres::CostFunction *newAutoDiffDifferenceCost(double weight);

ceres::CostFunction *newAutoDiffPointsDownwardsPrior(double weight);

} // namespace opencalibration
