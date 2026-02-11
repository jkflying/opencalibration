#include <opencalibration/ortho/blending.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

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

static cv::Mat fillInvalidRegions(const cv::Mat &color, const cv::Mat &weight)
{
    // Pull-push interpolation: extrapolate valid colors into zero-weight regions.
    // This prevents hard edges in the Laplacian pyramid that cause ringing artifacts
    // when all layers share the same validity boundary.
    int levels = 1;
    int min_dim = std::min(color.rows, color.cols);
    while ((min_dim >> levels) >= 2)
    {
        levels++;
    }

    cv::Mat weight3;
    std::vector<cv::Mat> w3ch = {weight, weight, weight};
    cv::merge(w3ch, weight3);

    std::vector<cv::Mat> wc_pyr(levels), w_pyr(levels);
    wc_pyr[0] = color.mul(weight3);
    w_pyr[0] = weight;

    for (int l = 1; l < levels; l++)
    {
        cv::pyrDown(wc_pyr[l - 1], wc_pyr[l]);
        cv::pyrDown(w_pyr[l - 1], w_pyr[l]);
    }

    // Coarsest level: normalize weighted color by weight
    cv::Mat w3_coarse;
    cv::merge(std::vector<cv::Mat>{w_pyr.back(), w_pyr.back(), w_pyr.back()}, w3_coarse);
    cv::Mat filled;
    cv::divide(wc_pyr.back(), cv::max(w3_coarse, 1e-6f), filled);

    // Pull back up: use original color where valid, upsampled fill where invalid
    for (int l = levels - 2; l >= 0; l--)
    {
        cv::Mat upsampled;
        cv::pyrUp(filled, upsampled, wc_pyr[l].size());

        cv::Mat w3_level;
        cv::merge(std::vector<cv::Mat>{w_pyr[l], w_pyr[l], w_pyr[l]}, w3_level);
        cv::Mat normalized;
        cv::divide(wc_pyr[l], cv::max(w3_level, 1e-6f), normalized);

        cv::Mat mask;
        cv::compare(w_pyr[l], 1e-6f, mask, cv::CMP_GT);

        filled = upsampled.clone();
        normalized.copyTo(filled, mask);
    }

    return filled;
}

cv::Mat laplacianBlend(const std::vector<cv::Mat> &lab_layers, const std::vector<cv::Mat> &weight_maps,
                       int pyramid_levels)
{
    if (lab_layers.empty())
    {
        return cv::Mat();
    }

    int rows = lab_layers[0].rows;
    int cols = lab_layers[0].cols;
    int num_layers = static_cast<int>(lab_layers.size());

    // Normalize weights so they sum to 1 per pixel
    std::vector<cv::Mat> norm_weights(num_layers);
    {
        cv::Mat weight_sum = cv::Mat::zeros(rows, cols, CV_32FC1);
        for (int i = 0; i < num_layers; i++)
        {
            weight_sum += weight_maps[i];
        }
        weight_sum = cv::max(weight_sum, 1e-6f);
        for (int i = 0; i < num_layers; i++)
        {
            cv::divide(weight_maps[i], weight_sum, norm_weights[i]);
        }
    }

    // Clamp pyramid levels to what the image dimensions allow
    int max_levels = 1;
    {
        int min_dim = std::min(rows, cols);
        while ((min_dim >> max_levels) >= 2 && max_levels < pyramid_levels)
        {
            max_levels++;
        }
    }
    pyramid_levels = max_levels;

    // Per-layer pull-push fill: each layer extrapolates from its own valid pixels.
    std::vector<cv::Mat> filled_layers(num_layers);
    for (int i = 0; i < num_layers; i++)
    {
        filled_layers[i] = fillInvalidRegions(lab_layers[i], norm_weights[i]);
    }

    // Build Gaussian pyramid for each weight map
    std::vector<std::vector<cv::Mat>> weight_pyramids(num_layers);
    for (int i = 0; i < num_layers; i++)
    {
        weight_pyramids[i].resize(pyramid_levels);
        weight_pyramids[i][0] = norm_weights[i];
        for (int l = 1; l < pyramid_levels; l++)
        {
            cv::pyrDown(weight_pyramids[i][l - 1], weight_pyramids[i][l]);
        }
    }

    // Renormalize weight pyramids at each level so they sum to 1.
    // pyrDown can break the partition-of-unity property at boundaries
    // where all layers share the same edge, causing darkening artifacts.
    for (int l = 1; l < pyramid_levels; l++)
    {
        cv::Mat level_sum = cv::Mat::zeros(weight_pyramids[0][l].size(), CV_32FC1);
        for (int i = 0; i < num_layers; i++)
        {
            level_sum += weight_pyramids[i][l];
        }
        level_sum = cv::max(level_sum, 1e-6f);
        for (int i = 0; i < num_layers; i++)
        {
            cv::divide(weight_pyramids[i][l], level_sum, weight_pyramids[i][l]);
        }
    }

    // Build Laplacian pyramid for each filled color layer
    std::vector<std::vector<cv::Mat>> color_pyramids(num_layers);
    for (int i = 0; i < num_layers; i++)
    {
        std::vector<cv::Mat> gaussian(pyramid_levels);
        gaussian[0] = filled_layers[i];
        for (int l = 1; l < pyramid_levels; l++)
        {
            cv::pyrDown(gaussian[l - 1], gaussian[l]);
        }

        color_pyramids[i].resize(pyramid_levels);
        for (int l = 0; l < pyramid_levels - 1; l++)
        {
            cv::Mat upsampled;
            cv::pyrUp(gaussian[l + 1], upsampled, gaussian[l].size());
            color_pyramids[i][l] = gaussian[l] - upsampled;
        }
        color_pyramids[i][pyramid_levels - 1] = gaussian[pyramid_levels - 1]; // coarsest level is Gaussian
    }

    // Blend at each pyramid level
    std::vector<cv::Mat> blended_pyramid(pyramid_levels);
    for (int l = 0; l < pyramid_levels; l++)
    {
        blended_pyramid[l] = cv::Mat::zeros(color_pyramids[0][l].size(), color_pyramids[0][l].type());
        for (int i = 0; i < num_layers; i++)
        {
            // Expand weight to 3 channels for multiplication
            cv::Mat weight3;
            std::vector<cv::Mat> channels = {weight_pyramids[i][l], weight_pyramids[i][l], weight_pyramids[i][l]};
            cv::merge(channels, weight3);
            blended_pyramid[l] += color_pyramids[i][l].mul(weight3);
        }
    }

    // Reconstruct from blended pyramid
    cv::Mat result = blended_pyramid[pyramid_levels - 1];
    for (int l = pyramid_levels - 2; l >= 0; l--)
    {
        cv::Mat upsampled;
        cv::pyrUp(result, upsampled, blended_pyramid[l].size());
        result = upsampled + blended_pyramid[l];
    }

    std::vector<cv::Mat> channels(3);
    cv::split(result, channels);
    channels[0] = cv::min(cv::max(channels[0], 0.0f), 100.0f);
    channels[1] = cv::min(cv::max(channels[1], -127.0f), 127.0f);
    channels[2] = cv::min(cv::max(channels[2], -127.0f), 127.0f);
    cv::Mat lab_clamped;
    cv::merge(channels, lab_clamped);

    cv::Mat bgr_float;
    cv::cvtColor(lab_clamped, bgr_float, cv::COLOR_Lab2BGR);

    cv::Mat bgr_result;
    bgr_float.convertTo(bgr_result, CV_8UC3, 255.0);

    // Add alpha channel (fully opaque for valid pixels)
    cv::Mat rgba;
    cv::cvtColor(bgr_result, rgba, cv::COLOR_BGR2BGRA);

    return rgba;
}

} // namespace opencalibration::orthomosaic
