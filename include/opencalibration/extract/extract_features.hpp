#pragma once

#include <opencalibration/types/feature_2d.hpp>

#include <opencv2/core/mat.hpp>

#include <vector>

namespace opencalibration
{

std::vector<feature_2d> extract_features(const cv::Mat &image);
}
