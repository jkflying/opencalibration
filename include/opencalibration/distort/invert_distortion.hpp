#pragma once

#include <opencalibration/types/camera_model.hpp>
#include <opencalibration/types/correspondence.hpp>
#include <opencalibration/types/feature_2d.hpp>
#include <opencalibration/types/feature_match.hpp>
#include <opencalibration/types/ray.hpp>

#include <eigen3/Eigen/Geometry>

#include <vector>

namespace opencalibration
{
InverseDistortionCameraModel<double> convertModel(const DifferentiableCameraModel<double> &standardModel);
DifferentiableCameraModel<double> convertModel(const InverseDistortionCameraModel<double> &invertedModel);
} // namespace opencalibration
