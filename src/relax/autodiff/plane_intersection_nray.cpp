#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <ceres/autodiff_cost_function.h>
#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{
namespace
{
template <int N, typename V> std::array<V, N> firstN(const std::vector<V> &values)
{
    std::array<V, N> first;
    std::copy_n(values.begin(), N, first.begin());
    return first;
}

template <int N> std::array<double, N> sigmasOrUnit(const std::vector<double> &inverse_sigmas)
{
    return inverse_sigmas.empty() ? unitSigmas<N>() : firstN<N>(inverse_sigmas);
}

template <typename F, int... PoseSizes> ceres::CostFunction *autoDiff(F *functor)
{
    return new ceres::AutoDiffCostFunction<F, F::NUM_RESIDUALS, 1, 1, 1, PoseSizes...>(functor);
}

template <int N>
PlaneIntersectionAngleCost_NRay<N> *newCost(const std::vector<Eigen::Vector3d> &camera_rays,
                                            const std::array<Eigen::Vector2d, 3> &plane_points,
                                            const std::vector<double> &inverse_sigmas)
{
    return new PlaneIntersectionAngleCost_NRay<N>(firstN<N>(camera_rays), plane_points,
                                                  sigmasOrUnit<N>(inverse_sigmas));
}

template <int N>
PlaneIntersectionAngleCost_NRay_FixedPositions<N> *newFixedPositionsCost(
    const std::vector<Eigen::Vector3d> &camera_rays, const std::array<Eigen::Vector2d, 3> &plane_points,
    const std::vector<Eigen::Vector3d> &camera_positions, const std::vector<double> &inverse_sigmas)
{
    return new PlaneIntersectionAngleCost_NRay_FixedPositions<N>(
        firstN<N>(camera_rays), plane_points, firstN<N>(camera_positions), sigmasOrUnit<N>(inverse_sigmas));
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
        return autoDiff<PlaneIntersectionAngleCost_NRay<3>, P, P, P>(
            newCost<3>(camera_rays, plane_points, inverse_sigmas));
    case 4:
        return autoDiff<PlaneIntersectionAngleCost_NRay<4>, P, P, P, P>(
            newCost<4>(camera_rays, plane_points, inverse_sigmas));
    case 5:
        return autoDiff<PlaneIntersectionAngleCost_NRay<5>, P, P, P, P, P>(
            newCost<5>(camera_rays, plane_points, inverse_sigmas));
    default:
        return nullptr;
    }
}

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost_NRay_FixedPositions(
    const std::vector<Eigen::Vector3d> &camera_rays, const std::array<Eigen::Vector2d, 3> &plane_points,
    const std::vector<Eigen::Vector3d> &camera_positions, const std::vector<double> &inverse_sigmas)
{
    constexpr int O = ORIENTATION_PARAMETERS;
    switch (camera_rays.size())
    {
    case 3:
        return autoDiff<PlaneIntersectionAngleCost_NRay_FixedPositions<3>, O, O, O>(
            newFixedPositionsCost<3>(camera_rays, plane_points, camera_positions, inverse_sigmas));
    case 4:
        return autoDiff<PlaneIntersectionAngleCost_NRay_FixedPositions<4>, O, O, O, O>(
            newFixedPositionsCost<4>(camera_rays, plane_points, camera_positions, inverse_sigmas));
    case 5:
        return autoDiff<PlaneIntersectionAngleCost_NRay_FixedPositions<5>, O, O, O, O, O>(
            newFixedPositionsCost<5>(camera_rays, plane_points, camera_positions, inverse_sigmas));
    default:
        return nullptr;
    }
}
} // namespace opencalibration
