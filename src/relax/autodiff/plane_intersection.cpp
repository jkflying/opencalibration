#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <ceres/autodiff_cost_function.h>
#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{
ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost(const Eigen::Vector3d &camera_ray1,
                                                           const Eigen::Vector3d &camera_ray2,
                                                           const Eigen::Vector2d &plane_point1,
                                                           const Eigen::Vector2d &plane_point2,
                                                           const Eigen::Vector2d &plane_point3,
                                                           const std::array<double, 2> &inverse_sigmas)
{
    using Functor = PlaneIntersectionAngleCost;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2, Functor::NUM_PARAMETERS_3,
                                                     Functor::NUM_PARAMETERS_4, Functor::NUM_PARAMETERS_5>;

    return new CostFunction(
        new Functor(camera_ray1, camera_ray2, plane_point1, plane_point2, plane_point3, inverse_sigmas));
}

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost_FixedPositions(
    const Eigen::Vector3d &camera_ray1, const Eigen::Vector3d &camera_ray2, const Eigen::Vector2d &plane_point1,
    const Eigen::Vector2d &plane_point2, const Eigen::Vector2d &plane_point3,
    const std::array<Eigen::Vector3d, 2> &camera_positions, const std::array<double, 2> &inverse_sigmas)
{
    using Functor = PlaneIntersectionAngleCost_FixedPositions;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2, Functor::NUM_PARAMETERS_3,
                                                     Functor::NUM_PARAMETERS_4, Functor::NUM_PARAMETERS_5>;
    return new CostFunction(new Functor(camera_ray1, camera_ray2, plane_point1, plane_point2, plane_point3,
                                        camera_positions, inverse_sigmas));
}
} // namespace opencalibration
