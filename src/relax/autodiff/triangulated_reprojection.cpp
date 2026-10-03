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

template <typename F, int BlockSize, size_t... I> ceres::CostFunction *autoDiff(F *functor, std::index_sequence<I...>)
{
    return new ceres::AutoDiffCostFunction<F, F::NUM_RESIDUALS, (static_cast<void>(I), BlockSize)...>(functor);
}

template <int N>
ceres::CostFunction *newCost(const std::vector<Eigen::Vector3d> &camera_rays, const std::vector<double> &inverse_sigmas)
{
    using F = TriangulatedReprojectionCost<N>;
    return autoDiff<F, POSE_PARAMETERS>(new F(firstN<N>(camera_rays), sigmasOrUnit<N>(inverse_sigmas)),
                                        std::make_index_sequence<N>{});
}

template <int N>
ceres::CostFunction *newFixedPositionsCost(const std::vector<Eigen::Vector3d> &camera_rays,
                                           const std::vector<Eigen::Vector3d> &camera_positions,
                                           const std::vector<double> &inverse_sigmas)
{
    using F = TriangulatedReprojectionCost_FixedPositions<N>;
    return autoDiff<F, ORIENTATION_PARAMETERS>(
        new F(firstN<N>(camera_rays), firstN<N>(camera_positions), sigmasOrUnit<N>(inverse_sigmas)),
        std::make_index_sequence<N>{});
}
} // namespace

ceres::CostFunction *newAutoDiffTriangulatedReprojectionCost(const std::vector<Eigen::Vector3d> &camera_rays,
                                                             const std::vector<double> &inverse_sigmas)
{
    switch (camera_rays.size())
    {
    case 2:
        return newCost<2>(camera_rays, inverse_sigmas);
    case 3:
        return newCost<3>(camera_rays, inverse_sigmas);
    case 4:
        return newCost<4>(camera_rays, inverse_sigmas);
    case 5:
        return newCost<5>(camera_rays, inverse_sigmas);
    default:
        return nullptr;
    }
}

ceres::CostFunction *newAutoDiffTriangulatedReprojectionCost_FixedPositions(
    const std::vector<Eigen::Vector3d> &camera_rays, const std::vector<Eigen::Vector3d> &camera_positions,
    const std::vector<double> &inverse_sigmas)
{
    switch (camera_rays.size())
    {
    case 2:
        return newFixedPositionsCost<2>(camera_rays, camera_positions, inverse_sigmas);
    case 3:
        return newFixedPositionsCost<3>(camera_rays, camera_positions, inverse_sigmas);
    case 4:
        return newFixedPositionsCost<4>(camera_rays, camera_positions, inverse_sigmas);
    case 5:
        return newFixedPositionsCost<5>(camera_rays, camera_positions, inverse_sigmas);
    default:
        return nullptr;
    }
}
} // namespace opencalibration
