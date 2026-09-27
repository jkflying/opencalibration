#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <ceres/autodiff_cost_function.h>
#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{
ceres::CostFunction *newAutoDiffPixelErrorCost_Orientation(const CameraModel &camera_model,
                                                           const Eigen::Vector2d &camera_pixel)
{
    using Functor = PixelErrorCost_Orientation;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2>;
    return new CostFunction(new Functor(camera_model, camera_pixel));
}
} // namespace opencalibration
