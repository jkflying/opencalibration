#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <opencalibration/relax/padded_autodiff_cost_function.hpp>
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

template <typename F, int... PoseSizes> ceres::CostFunction *autoDiff(F *functor)
{
    return new PaddedAutoDiffCostFunction<F, F::NUM_RESIDUALS, 1, 1, 1, 1, 2, 3, PoseSizes...>(functor);
}

template <int N>
PlaneIntersectionAngleCost_NRay_FocalRadial<N> *newCost(const std::vector<Eigen::Vector2d> &camera_pixels,
                                                        const std::array<Eigen::Vector2d, 3> &plane_points,
                                                        const InverseDifferentiableCameraModel<double> &model)
{
    return new PlaneIntersectionAngleCost_NRay_FocalRadial<N>(firstN<N>(camera_pixels), plane_points, model);
}

template <int N>
PlaneIntersectionAngleCost_NRay_FocalRadial_FixedPositions<N> *newFixedPositionsCost(
    const std::vector<Eigen::Vector2d> &camera_pixels, const std::array<Eigen::Vector2d, 3> &plane_points,
    const InverseDifferentiableCameraModel<double> &model, const std::vector<Eigen::Vector3d> &camera_positions)
{
    return new PlaneIntersectionAngleCost_NRay_FocalRadial_FixedPositions<N>(firstN<N>(camera_pixels), plane_points,
                                                                             model, firstN<N>(camera_positions));
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
        return autoDiff<PlaneIntersectionAngleCost_NRay_FocalRadial<3>, P, P, P>(
            newCost<3>(camera_pixels, plane_points, model));
    case 4:
        return autoDiff<PlaneIntersectionAngleCost_NRay_FocalRadial<4>, P, P, P, P>(
            newCost<4>(camera_pixels, plane_points, model));
    case 5:
        return autoDiff<PlaneIntersectionAngleCost_NRay_FocalRadial<5>, P, P, P, P, P>(
            newCost<5>(camera_pixels, plane_points, model));
    default:
        return nullptr;
    }
}

ceres::CostFunction *newAutoDiffPlaneIntersectionAngleCost_NRay_FocalRadial_FixedPositions(
    const std::vector<Eigen::Vector2d> &camera_pixels, const std::array<Eigen::Vector2d, 3> &plane_points,
    const InverseDifferentiableCameraModel<double> &model, const std::vector<Eigen::Vector3d> &camera_positions)
{
    constexpr int O = ORIENTATION_PARAMETERS;
    switch (camera_pixels.size())
    {
    case 3:
        return autoDiff<PlaneIntersectionAngleCost_NRay_FocalRadial_FixedPositions<3>, O, O, O>(
            newFixedPositionsCost<3>(camera_pixels, plane_points, model, camera_positions));
    case 4:
        return autoDiff<PlaneIntersectionAngleCost_NRay_FocalRadial_FixedPositions<4>, O, O, O, O>(
            newFixedPositionsCost<4>(camera_pixels, plane_points, model, camera_positions));
    case 5:
        return autoDiff<PlaneIntersectionAngleCost_NRay_FocalRadial_FixedPositions<5>, O, O, O, O, O>(
            newFixedPositionsCost<5>(camera_pixels, plane_points, model, camera_positions));
    default:
        return nullptr;
    }
}
} // namespace opencalibration
