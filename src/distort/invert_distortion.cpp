#include <opencalibration/distort/invert_distortion.hpp>

namespace opencalibration
{

DifferentiableCameraModel<double> standard2InvertedModel(const DifferentiableCameraModel<double> &standardModel)
{
    DifferentiableCameraModel<double> inverted = standardModel;

    // TODO: invert the distortion variables so that the model can be used in inverse projections differentiably

    return inverted;
}

DifferentiableCameraModel<double> inverted2StandardModel(const DifferentiableCameraModel<double> &invertedModel)
{
    DifferentiableCameraModel<double> standard = invertedModel;

    // TODO: invert the distortion variables so that the model can be used in standard projections differentiably

    return standard;
}

} // namespace opencalibration
