#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <ceres/autodiff_cost_function.h>
#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{
namespace
{
template <int N, int... PoseSizes>
ceres::CostFunction *makeMultiRayCostFocalRadialImpl(const std::vector<Eigen::Vector2d> &camera_pixels,
                                                     const std::array<Eigen::Vector2d, 3> &plane_points,
                                                     const InverseDifferentiableCameraModel<double> &model)
{
    using F = PlaneIntersectionAngleCost_NRay_FocalRadial<N>;
    std::array<Eigen::Vector2d, N> pixels;
    for (int i = 0; i < N; i++)
    {
        pixels[i] = camera_pixels[i];
    }
    return new ceres::AutoDiffCostFunction<F, F::NUM_RESIDUALS, 1, 1, 1, 1, 2, 3, PoseSizes...>(
        new F(pixels, plane_points, model));
}
} // namespace

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost_NRay_FocalRadial(
    const std::vector<Eigen::Vector2d> &camera_pixels, const std::array<Eigen::Vector2d, 3> &plane_points,
    const InverseDifferentiableCameraModel<double> &model)
{
    constexpr int P = POSE_PARAMETERS;
    switch (camera_pixels.size())
    {
    case 3:
        return makeMultiRayCostFocalRadialImpl<3, P, P, P>(camera_pixels, plane_points, model);
    case 4:
        return makeMultiRayCostFocalRadialImpl<4, P, P, P, P>(camera_pixels, plane_points, model);
    case 5:
        return makeMultiRayCostFocalRadialImpl<5, P, P, P, P, P>(camera_pixels, plane_points, model);
    default:
        return nullptr;
    }
}
} // namespace opencalibration
