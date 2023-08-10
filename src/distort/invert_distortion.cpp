#include <opencalibration/distort/invert_distortion.hpp>

namespace opencalibration
{

InverseDifferentiableCameraModel<double> convertModel(const DifferentiableCameraModel<double> &standardModel)
{
    InverseDifferentiableCameraModel<double> inverted = standardModel.cast<double, CameraModelTag::INVERSE>();
    inverted.radial_distortion = {};
    inverted.tangential_distortion = {};

    // TODO: invert the distortion variables so that the model can be used in inverse projections differentiably

    return inverted;
}

DifferentiableCameraModel<double> convertModel(const InverseDifferentiableCameraModel<double> &invertedModel)
{
    DifferentiableCameraModel<double> standard = invertedModel.cast<double, CameraModelTag::FORWARD>();

    standard.radial_distortion = {};
    standard.tangential_distortion = {};
    // TODO: invert the distortion variables so that the model can be used in standard projections differentiably

    return standard;
}

} // namespace opencalibration
