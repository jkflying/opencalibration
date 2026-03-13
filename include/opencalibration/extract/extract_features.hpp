#pragma once

#include <opencalibration/types/feature_2d.hpp>

#include <opencv2/core/mat.hpp>

#include <vector>

namespace opencalibration
{

struct extracted_features
{
    std::vector<feature_2d> features;
    size_t num_sparse_features = 0;
};

extracted_features extract_features(const cv::Mat &image);
} // namespace opencalibration
