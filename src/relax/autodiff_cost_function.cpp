#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <ceres/autodiff_cost_function.h>
#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{
ceres::CostFunction *newAutoDiffMultiDecomposedRotationCost(const camera_relations &relations,
                                                            const Eigen::Vector3d *translation1,
                                                            const Eigen::Vector3d *translation2)

{
    using Functor = MultiDecomposedRotationCost;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2>;
    return new CostFunction(new Functor(relations, translation1, translation2));
}

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost(
    const Eigen::Vector3d &camera_loc1, const Eigen::Vector3d &camera_loc2, const Eigen::Vector3d &camera_ray1,
    const Eigen::Vector3d &camera_ray2, const Eigen::Vector2d &plane_point1, const Eigen::Vector2d &plane_point2,
    const Eigen::Vector2d &plane_point3)
{
    using Functor = PlaneIntersectionAngleCost;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2, Functor::NUM_PARAMETERS_3,
                                                     Functor::NUM_PARAMETERS_4, Functor::NUM_PARAMETERS_5>;

    return new CostFunction(
        new Functor(camera_loc1, camera_loc2, camera_ray1, camera_ray2, plane_point1, plane_point2, plane_point3));
}

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost_FocalRadial(
    const Eigen::Vector3d &camera_loc1, const Eigen::Vector3d &camera_loc2, const Eigen::Vector2d &camera_pixel1,
    const Eigen::Vector2d &camera_pixel2, const Eigen::Vector2d &plane_point1, const Eigen::Vector2d &plane_point2,
    const Eigen::Vector2d &plane_point3, const InverseDifferentiableCameraModel<double> &model)
{
    using Functor = PlaneIntersectionAngleCost_OrientationFocalRadial_SharedModel;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2, Functor::NUM_PARAMETERS_3,
                                                     Functor::NUM_PARAMETERS_4, Functor::NUM_PARAMETERS_5,
                                                     Functor::NUM_PARAMETERS_6, Functor::NUM_PARAMETERS_7>;

    return new CostFunction(
        new Functor(camera_loc1, camera_loc2, camera_pixel1, camera_pixel2, plane_point1, plane_point2, plane_point3, model));
}

ceres::CostFunction *newAutoDiffPixelErrorCost_Orientation(const Eigen::Vector3d &camera_loc,
                                                           const CameraModel &camera_model,
                                                           const Eigen::Vector2d &camera_pixel)
{
    using Functor = PixelErrorCost_Orientation;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2>;
    return new CostFunction(new Functor(camera_loc, camera_model, camera_pixel));
}

ceres::CostFunction *newAutoDiffPixelErrorCost_OrientationFocal(const Eigen::Vector3d &camera_loc,
                                                                const CameraModel &camera_model,
                                                                const Eigen::Vector2d &camera_pixel)
{
    using Functor = PixelErrorCost_OrientationFocal;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2, Functor::NUM_PARAMETERS_3>;
    return new CostFunction(new Functor(camera_loc, camera_model, camera_pixel));
}

ceres::CostFunction *newAutoDiffPixelErrorCost_OrientationFocalRadial(const Eigen::Vector3d &camera_loc,
                                                                      const CameraModel &camera_model,
                                                                      const Eigen::Vector2d &camera_pixel)
{
    using Functor = PixelErrorCost_OrientationFocalRadial;
    using CostFunction =
        ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                    Functor::NUM_PARAMETERS_2, Functor::NUM_PARAMETERS_3, Functor::NUM_PARAMETERS_4>;
    return new CostFunction(new Functor(camera_loc, camera_model, camera_pixel));
}

ceres::CostFunction *newAutoDiffPixelErrorCost_OrientationFocalRadialTangential(const Eigen::Vector3d &camera_loc,
                                                                                const CameraModel &camera_model,
                                                                                const Eigen::Vector2d &camera_pixel)
{
    using Functor = PixelErrorCost_OrientationFocalRadialTangential;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2, Functor::NUM_PARAMETERS_3,
                                                     Functor::NUM_PARAMETERS_4, Functor::NUM_PARAMETERS_5>;
    return new CostFunction(new Functor(camera_loc, camera_model, camera_pixel));
}

ceres::CostFunction *newAutoDiffDifferenceCost(double weight)
{
    using Functor = DifferenceCost;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2>;
    return new CostFunction(new Functor(weight));
}

ceres::CostFunction *newAutoDiffPointsDownwardsPrior(double weight)
{
    using Functor = PointsDownwardsPrior;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1>;
    return new CostFunction(new Functor(weight));
}
} // namespace opencalibration
