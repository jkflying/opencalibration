#pragma once

#include <opencalibration/types/feature_2d.hpp>

#include <opencv2/core/mat.hpp>

#include <vector>

namespace opencalibration
{

constexpr int FEATURE_MAX_LENGTH_PIXELS = 1600;

inline Eigen::Vector2d unscale_pixel(const Eigen::Vector2d &scaled_pixel, double scale)
{
    return (scaled_pixel.array() + 0.5) / scale - 0.5;
}

struct extracted_features
{
    std::vector<feature_2d> features;
    size_t num_sparse_features = 0;
};

extracted_features extract_features(const cv::Mat &image);
} // namespace opencalibration
