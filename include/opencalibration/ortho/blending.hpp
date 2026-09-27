#pragma once

namespace opencalibration::orthomosaic
{

float computeBlendWeight(float pixel_x, float pixel_y, int image_width, int image_height, float camera_distance);

} // namespace opencalibration::orthomosaic
