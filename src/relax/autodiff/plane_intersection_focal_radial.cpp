#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <ceres/autodiff_cost_function.h>
#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{
ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost_FocalRadial(
    const Eigen::Vector2d &camera_pixel1, const Eigen::Vector2d &camera_pixel2, const Eigen::Vector2d &plane_point1,
    const Eigen::Vector2d &plane_point2, const Eigen::Vector2d &plane_point3,
    const InverseDifferentiableCameraModel<double> &model)
{
    using Functor = PlaneIntersectionAngleCost_OrientationFocalRadial_SharedModel;
    using CostFunction =
        ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                    Functor::NUM_PARAMETERS_2, Functor::NUM_PARAMETERS_3, Functor::NUM_PARAMETERS_4,
                                    Functor::NUM_PARAMETERS_5, Functor::NUM_PARAMETERS_6, Functor::NUM_PARAMETERS_7,
                                    Functor::NUM_PARAMETERS_8>;

    return new CostFunction(new Functor(camera_pixel1, camera_pixel2, plane_point1, plane_point2, plane_point3, model));
}

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost_FocalRadial_FixedPositions(
    const Eigen::Vector2d &camera_pixel1, const Eigen::Vector2d &camera_pixel2, const Eigen::Vector2d &plane_point1,
    const Eigen::Vector2d &plane_point2, const Eigen::Vector2d &plane_point3,
    const InverseDifferentiableCameraModel<double> &model, const std::array<Eigen::Vector3d, 2> &camera_positions)
{
    using Functor = PlaneIntersectionAngleCost_OrientationFocalRadial_SharedModel_FixedPositions;
    using CostFunction =
        ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                    Functor::NUM_PARAMETERS_2, Functor::NUM_PARAMETERS_3, Functor::NUM_PARAMETERS_4,
                                    Functor::NUM_PARAMETERS_5, Functor::NUM_PARAMETERS_6, Functor::NUM_PARAMETERS_7,
                                    Functor::NUM_PARAMETERS_8>;
    return new CostFunction(
        new Functor(camera_pixel1, camera_pixel2, plane_point1, plane_point2, plane_point3, model, camera_positions));
}
} // namespace opencalibration
