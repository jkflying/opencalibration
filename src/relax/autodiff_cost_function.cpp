#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <ceres/autodiff_cost_function.h>
#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{
ceres::CostFunction *newAutoDiffMultiDecomposedRotationCost(const camera_relations &relations,
                                                            const Eigen::Vector3d *translation1,
                                                            const Eigen::Vector3d *translation2)

{

    using CostFunction =
        ceres::AutoDiffCostFunction<MultiDecomposedRotationCost, MultiDecomposedRotationCost::NUM_RESIDUALS,
                                    MultiDecomposedRotationCost::NUM_PARAMETERS_1,
                                    MultiDecomposedRotationCost::NUM_PARAMETERS_2>;
    return new CostFunction(new MultiDecomposedRotationCost(relations, translation1, translation2));
}

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost(
    const Eigen::Vector3d &camera_loc1, const Eigen::Vector3d &camera_loc2, const Eigen::Vector3d &camera_ray1,
    const Eigen::Vector3d &camera_ray2, const Eigen::Vector2d &plane_point1, const Eigen::Vector2d &plane_point2,
    const Eigen::Vector2d &plane_point3)
{
    using CostFunction = ceres::AutoDiffCostFunction<
        PlaneIntersectionAngleCost, PlaneIntersectionAngleCost::NUM_RESIDUALS,
        PlaneIntersectionAngleCost::NUM_PARAMETERS_1, PlaneIntersectionAngleCost::NUM_PARAMETERS_2,
        PlaneIntersectionAngleCost::NUM_PARAMETERS_3, PlaneIntersectionAngleCost::NUM_PARAMETERS_4,
        PlaneIntersectionAngleCost::NUM_PARAMETERS_5>;

    return new CostFunction(new PlaneIntersectionAngleCost(camera_loc1, camera_loc2, camera_ray1, camera_ray2,
                                                           plane_point1, plane_point2, plane_point3));
}

ceres::CostFunction *newAutoDiffPixelErrorCost(const Eigen::Vector3d &camera_loc, const CameraModel &camera_model,
                                               const Eigen::Vector2d &camera_pixel)
{
    using CostFunction =
        ceres::AutoDiffCostFunction<PixelErrorCost, PixelErrorCost::NUM_RESIDUALS, PixelErrorCost::NUM_PARAMETERS_1,
                                    PixelErrorCost::NUM_PARAMETERS_2>;
    return new CostFunction(new PixelErrorCost(camera_loc, camera_model, camera_pixel));
}

ceres::CostFunction *newAutoDiffPointsDownwardsPrior()
{
    using CostFunction = ceres::AutoDiffCostFunction<PointsDownwardsPrior, PointsDownwardsPrior::NUM_RESIDUALS,
                                                     PointsDownwardsPrior::NUM_PARAMETERS_1>;
    return new CostFunction(new PointsDownwardsPrior());
}
} // namespace opencalibration
