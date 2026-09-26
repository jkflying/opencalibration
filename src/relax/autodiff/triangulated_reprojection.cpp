#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <ceres/autodiff_cost_function.h>
#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{
namespace
{
template <int N, int... PoseSizes>
ceres::CostFunction *makeTriangulatedReprojectionCost(const std::vector<Eigen::Vector3d> &camera_rays,
                                                      const std::vector<Eigen::Vector3d> &camera_positions)
{
    using F = TriangulatedReprojectionCost<N>;
    std::array<Eigen::Vector3d, N> rays;
    std::copy_n(camera_rays.begin(), N, rays.begin());
    if (camera_positions.empty())
        return new ceres::AutoDiffCostFunction<F, F::NUM_RESIDUALS, PoseSizes...>(new F(rays));
    std::array<Eigen::Vector3d, N> positions;
    std::copy_n(camera_positions.begin(), N, positions.begin());
    return new ceres::AutoDiffCostFunction<F, F::NUM_RESIDUALS, PoseSizes...>(new F(rays, positions));
}

template <int N, int P, size_t... I>
ceres::CostFunction *makeWithPoseSize(const std::vector<Eigen::Vector3d> &camera_rays,
                                      const std::vector<Eigen::Vector3d> &camera_positions, std::index_sequence<I...>)
{
    return makeTriangulatedReprojectionCost<N, (static_cast<void>(I), P)...>(camera_rays, camera_positions);
}

template <int N>
ceres::CostFunction *make(const std::vector<Eigen::Vector3d> &camera_rays,
                          const std::vector<Eigen::Vector3d> &camera_positions)
{
    if (camera_positions.empty())
        return makeWithPoseSize<N, POSE_PARAMETERS>(camera_rays, camera_positions, std::make_index_sequence<N>{});
    return makeWithPoseSize<N, 4>(camera_rays, camera_positions, std::make_index_sequence<N>{});
}
} // namespace

ceres::CostFunction *newAutoDiffTriangulatedReprojectionCost(const std::vector<Eigen::Vector3d> &camera_rays,
                                                             const std::vector<Eigen::Vector3d> &camera_positions)
{
    switch (camera_rays.size())
    {
    case 2:
        return make<2>(camera_rays, camera_positions);
    case 3:
        return make<3>(camera_rays, camera_positions);
    case 4:
        return make<4>(camera_rays, camera_positions);
    case 5:
        return make<5>(camera_rays, camera_positions);
    default:
        return nullptr;
    }
}
} // namespace opencalibration
