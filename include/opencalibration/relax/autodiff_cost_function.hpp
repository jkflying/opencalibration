#pragma once

#include <opencalibration/types/camera_model.hpp>
#include <opencalibration/types/camera_relations.hpp>

#include <ceres/cost_function.h>
#include <eigen3/Eigen/Core>

#include <array>
#include <vector>

namespace opencalibration
{
ceres::CostFunction *newAutoDiffMultiDecomposedRotationCost(const camera_relations &relations);

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost(const Eigen::Vector3d &camera_ray1,
                                                           const Eigen::Vector3d &camera_ray2,
                                                           const Eigen::Vector2d &plane_point1,
                                                           const Eigen::Vector2d &plane_point2,
                                                           const Eigen::Vector2d &plane_point3,
                                                           const std::array<double, 2> &inverse_sigmas = {1, 1});

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost_FocalRadial(
    const Eigen::Vector2d &camera_pixel1, const Eigen::Vector2d &camera_pixel2, const Eigen::Vector2d &plane_point1,
    const Eigen::Vector2d &plane_point2, const Eigen::Vector2d &plane_point3,
    const InverseDifferentiableCameraModel<double> &model);

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost_NRay(const std::vector<Eigen::Vector3d> &camera_rays,
                                                                const std::array<Eigen::Vector2d, 3> &plane_points,
                                                                const std::vector<double> &inverse_sigmas = {});

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost_NRay_FocalRadial(
    const std::vector<Eigen::Vector2d> &camera_pixels, const std::array<Eigen::Vector2d, 3> &plane_points,
    const InverseDifferentiableCameraModel<double> &model);

ceres::CostFunction *newAutoDiffTriangulatedReprojectionCost(const std::vector<Eigen::Vector3d> &camera_rays,
                                                             const std::vector<Eigen::Vector3d> &camera_positions = {},
                                                             const std::vector<double> &inverse_sigmas = {});

ceres::CostFunction *newAutoDiffPixelErrorCost_Orientation(const CameraModel &camera_model,
                                                           const Eigen::Vector2d &camera_pixel);

ceres::CostFunction *newAutoDiffPixelErrorCost_OrientationFocal(const CameraModel &camera_model,
                                                                const Eigen::Vector2d &camera_pixel);

ceres::CostFunction *newAutoDiffPixelErrorCost_OrientationFocalRadial(const CameraModel &camera_model,
                                                                      const Eigen::Vector2d &camera_pixel);

ceres::CostFunction *newAutoDiffPixelErrorCost_OrientationFocalRadialTangential(const CameraModel &camera_model,
                                                                                const Eigen::Vector2d &camera_pixel);
ceres::CostFunction *newAutoDiffDifferenceCost(double weight);

ceres::CostFunction *newAutoDiffValuePrior(double target, double weight);

ceres::CostFunction *newAutoDiffPointsDownwardsPrior(double weight);

ceres::CostFunction *newAutoDiffGPSPositionPrior(const Eigen::Vector3d &gps_position, double horizontal_weight,
                                                 double vertical_weight);

ceres::CostFunction *newAutoDiffDistortionMonotonicityCost(double r_max, double weight);

ceres::CostFunction *newAutoDiffAdjacentTriangleNormalCost(const Eigen::Vector2d &xyA, const Eigen::Vector2d &xyB,
                                                           const Eigen::Vector2d &xyC, const Eigen::Vector2d &xyD,
                                                           double weight);

ceres::CostFunction *newAutoDiffMeshPointHeightCost(const Eigen::Vector3d &barycentric, double z, double weight);

} // namespace opencalibration
