#include <opencalibration/ortho/blending.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace opencalibration::orthomosaic
{

void rejectOutliers(LayeredTileBuffer &tile, double mad_threshold)
{
    for (int row = 0; row < tile.height; row++)
    {
        for (int col = 0; col < tile.width; col++)
        {
            std::vector<float> L_values;
            std::vector<int> valid_layers;
            for (int layer = 0; layer < tile.num_layers; layer++)
            {
                const auto &sample = tile.at(layer, row, col);
                if (sample.valid)
                {
                    cv::Mat bgr_pixel(1, 1, CV_8UC3);
                    bgr_pixel.at<cv::Vec3b>(0, 0) = sample.color_bgr;
                    cv::Mat lab_pixel;
                    cv::cvtColor(bgr_pixel, lab_pixel, cv::COLOR_BGR2Lab);
                    L_values.push_back(lab_pixel.at<cv::Vec3b>(0, 0)[0]);
                    valid_layers.push_back(layer);
                }
            }

            if (valid_layers.size() < 3)
            {
                continue; // Need at least 3 samples for meaningful outlier rejection
            }

            std::vector<float> sorted_L = L_values;
            std::sort(sorted_L.begin(), sorted_L.end());
            float median = sorted_L[sorted_L.size() / 2];

            // Compute MAD (median absolute deviation)
            std::vector<float> abs_devs;
            abs_devs.reserve(L_values.size());
            for (float l : L_values)
            {
                abs_devs.push_back(std::abs(l - median));
            }
            std::sort(abs_devs.begin(), abs_devs.end());
            float mad = abs_devs[abs_devs.size() / 2] * 1.4826f; // scale factor for normal distribution

            if (mad < 1.0f)
            {
                continue; // Very consistent values, no outliers
            }

            for (size_t i = 0; i < valid_layers.size(); i++)
            {
                if (std::abs(L_values[i] - median) > mad_threshold * mad)
                {
                    tile.at(valid_layers[i], row, col).valid = false;
                }
            }
        }
    }
}

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

    // Build Laplacian pyramid for each color layer
    std::vector<std::vector<cv::Mat>> color_pyramids(num_layers);
    for (int i = 0; i < num_layers; i++)
    {
        std::vector<cv::Mat> gaussian(pyramid_levels);
        gaussian[0] = lab_layers[i];
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

    // Clamp to valid LAB range and convert to BGR uint8
    // LAB float32: L in [0,100], a in [-127,127], b in [-127,127]
    // But OpenCV's 8-bit LAB: L in [0,255], a in [0,255], b in [0,255] (with 128 offset)
    cv::Mat lab_clamped;
    result.convertTo(lab_clamped, CV_8UC3);

    cv::Mat bgr_result;
    cv::cvtColor(lab_clamped, bgr_result, cv::COLOR_Lab2BGR);

    // Add alpha channel (fully opaque for valid pixels)
    cv::Mat rgba;
    cv::cvtColor(bgr_result, rgba, cv::COLOR_BGR2BGRA);

    return rgba;
}

} // namespace opencalibration::orthomosaic
