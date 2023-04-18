#include <opencalibration/distort/invert_distortion.hpp>

namespace opencalibration
{

InverseDistortionCameraModel<double> convertModel(const DifferentiableCameraModel<double> &standardModel)
{
    InverseDistortionCameraModel<double> inverted = standardModel;
    inverted.radial_distortion = {};
    inverted.tangential_distortion = {};

    // TODO: invert the distortion variables so that the model can be used in inverse projections differentiably

    return inverted;
}

DifferentiableCameraModel<double> convertModel(const InverseDistortionCameraModel<double> &invertedModel)
{
    DifferentiableCameraModel<double> standard = invertedModel;

    standard.radial_distortion = {};
    standard.tangential_distortion = {};
    // TODO: invert the distortion variables so that the model can be used in standard projections differentiably

    return standard;
}

} // namespace opencalibration
