#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <ceres/autodiff_cost_function.h>
#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{
ceres::CostFunction *newAutoDiffMultiDecomposedRotationCost(const camera_relations &relations)
{
    using Functor = MultiDecomposedRotationCost;
    using CostFunction = ceres::AutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2>;
    return new CostFunction(new Functor(relations));
}
} // namespace opencalibration
