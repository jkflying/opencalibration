#pragma once

#include <opencalibration/types/feature_2d.hpp>

#include <opencv2/core/mat.hpp>

#include <vector>

namespace opencalibration
{

struct extracted_features
{
    std::vector<feature_2d> features;
    std::vector<feature_2d> dense_features;
};

extracted_features extract_features(const cv::Mat &image);
} // namespace opencalibration
