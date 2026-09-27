#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <ceres/autodiff_cost_function.h>
#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{
namespace
{
template <int N, int... PoseSizes>
ceres::CostFunction *makeMultiRayCostImpl(const std::vector<Eigen::Vector3d> &camera_rays,
                                          const std::array<Eigen::Vector2d, 3> &plane_points,
                                          const std::vector<double> &inverse_sigmas)
{
    using F = PlaneIntersectionAngleCost_NRay<N>;
    std::array<Eigen::Vector3d, N> dirs;
    std::array<double, N> sigmas = unitSigmas<N>();
    for (int i = 0; i < N; i++)
    {
        dirs[i] = camera_rays[i];
        if (!inverse_sigmas.empty())
            sigmas[i] = inverse_sigmas[i];
    }
    return new ceres::AutoDiffCostFunction<F, F::NUM_RESIDUALS, 1, 1, 1, PoseSizes...>(
        new F(dirs, plane_points, sigmas));
}
} // namespace

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost_NRay(const std::vector<Eigen::Vector3d> &camera_rays,
                                                                const std::array<Eigen::Vector2d, 3> &plane_points,
                                                                const std::vector<double> &inverse_sigmas)
{
    constexpr int P = POSE_PARAMETERS;
    switch (camera_rays.size())
    {
    case 3:
        return makeMultiRayCostImpl<3, P, P, P>(camera_rays, plane_points, inverse_sigmas);
    case 4:
        return makeMultiRayCostImpl<4, P, P, P, P>(camera_rays, plane_points, inverse_sigmas);
    case 5:
        return makeMultiRayCostImpl<5, P, P, P, P, P>(camera_rays, plane_points, inverse_sigmas);
    default:
        return nullptr;
    }
}
} // namespace opencalibration
