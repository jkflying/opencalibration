#include <opencalibration/ortho/blending.hpp>

#include <algorithm>
#include <cmath>

namespace opencalibration::orthomosaic
{

float computeBlendWeight(float pixel_x, float pixel_y, int image_width, int image_height, float camera_distance)
{
    float half_w = image_width * 0.5f;
    float half_h = image_height * 0.5f;

    // Edge distance weight: feather near image borders
    float dist_to_left = pixel_x;
    float dist_to_right = image_width - 1.0f - pixel_x;
    float dist_to_top = pixel_y;
    float dist_to_bottom = image_height - 1.0f - pixel_y;
    float min_edge_dist = std::min({dist_to_left, dist_to_right, dist_to_top, dist_to_bottom});
    float edge_weight = std::min(min_edge_dist / half_w, 1.0f);
    edge_weight = std::max(edge_weight, 0.001f); // small epsilon to avoid zero weights

    // Center distance weight: prefer pixels near image center
    float cx = (pixel_x - half_w) / half_w;
    float cy = (pixel_y - half_h) / half_h;
    float center_dist = std::sqrt(cx * cx + cy * cy);
    float center_weight = 1.0f - 0.5f * std::min(center_dist, 1.0f);

    // Proximity weight: prefer closer cameras
    float proximity_weight = 1.0f / (1.0f + camera_distance * camera_distance);

    return edge_weight * center_weight * proximity_weight;
}

} // namespace opencalibration::orthomosaic
