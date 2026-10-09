#include <opencalibration/relax/autodiff_cost_function.hpp>

#include <opencalibration/relax/padded_autodiff_cost_function.hpp>
#include <opencalibration/relax/relax_cost_function.hpp>

namespace opencalibration
{
ceres::CostFunction *newAutoDiffMultiDecomposedRotationCost(const camera_relations &relations)
{
    using Functor = MultiDecomposedRotationCost;
    using CostFunction = PaddedAutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2>;
    return new CostFunction(new Functor(relations));
}

ceres::CostFunction *newAutoDiffMultiDecomposedRotationCost_FixedPositions(const camera_relations &relations,
                                                                           const Eigen::Vector3d &dest_minus_source)
{
    using Functor = MultiDecomposedRotationCost_FixedPositions;
    using CostFunction = PaddedAutoDiffCostFunction<Functor, Functor::NUM_RESIDUALS, Functor::NUM_PARAMETERS_1,
                                                     Functor::NUM_PARAMETERS_2>;
    return new CostFunction(new Functor(relations, dest_minus_source));
}
} // namespace opencalibration
