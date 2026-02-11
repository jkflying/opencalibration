#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace opencalibration::orthomosaic
{

struct PixelSample
{
    cv::Vec3b color_bgr;
    size_t camera_id = 0;
    uint32_t model_id = 0;
    float normalized_radius = 0; // distance from image center, normalized to [0,1]
    float view_angle = 0;        // angle between camera direction and surface normal (radians)
    float weight = 0;
    bool valid = false;
};

struct LayeredTileBuffer
{
    int width = 0;
    int height = 0;
    int num_layers = 0;
    // layers[layer_idx][row * width + col]
    std::vector<std::vector<PixelSample>> layers;

    void resize(int w, int h, int n)
    {
        width = w;
        height = h;
        num_layers = n;
        layers.resize(n);
        for (auto &layer : layers)
        {
            layer.resize(w * h);
        }
    }

    PixelSample &at(int layer, int row, int col)
    {
        return layers[layer][row * width + col];
    }

    const PixelSample &at(int layer, int row, int col) const
    {
        return layers[layer][row * width + col];
    }
};

float computeBlendWeight(float pixel_x, float pixel_y, int image_width, int image_height, float camera_distance);

cv::Mat laplacianBlend(const std::vector<cv::Mat> &lab_layers, const std::vector<cv::Mat> &weight_maps,
                       int pyramid_levels);

} // namespace opencalibration::orthomosaic
