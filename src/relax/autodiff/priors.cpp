#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <ceres/autodiff_cost_function.h>
#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{
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

ceres::CostFunction *newAutoDiffGPSPositionPrior(const Eigen::Vector3d &gps_position, double horizontal_weight,
                                                 double vertical_weight)
{
    using Functor = GPSPositionPrior;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1>;
    return new CostFunction(new Functor(gps_position, horizontal_weight, vertical_weight));
}

ceres::CostFunction *newAutoDiffDistortionMonotonicityCost(double r_max, double weight)
{
    using Functor = DistortionMonotonicityCost;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1>;
    return new CostFunction(new Functor(r_max, weight));
}
ceres::CostFunction *newAutoDiffAdjacentTriangleNormalCost(const Eigen::Vector2d &xyA, const Eigen::Vector2d &xyB,
                                                           const Eigen::Vector2d &xyC, const Eigen::Vector2d &xyD,
                                                           double weight)
{
    using Functor = AdjacentTriangleNormalCost;
    using CostFunction =
        ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                    Functor::NUM_PARAMETERS_2, Functor::NUM_PARAMETERS_3, Functor::NUM_PARAMETERS_4>;
    return new CostFunction(new Functor(xyA, xyB, xyC, xyD, weight));
}

ceres::CostFunction *newAutoDiffMeshPointHeightCost(const Eigen::Vector3d &barycentric, double z, double weight)
{
    using Functor = MeshPointHeightCost;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2, Functor::NUM_PARAMETERS_3>;
    return new CostFunction(new Functor(barycentric, z, weight));
}
} // namespace opencalibration
