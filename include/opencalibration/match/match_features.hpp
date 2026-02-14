#pragma once

#include <opencalibration/types/feature_2d.hpp>
#include <opencalibration/types/feature_match.hpp>

#include <eigen3/Eigen/Core>

#include <tuple>
#include <vector>

namespace opencalibration
{
std::vector<size_t> spatially_subsample_feature_indices(const std::vector<feature_2d> &features, double spacing_pixels);

std::vector<feature_match> match_features(const std::vector<feature_2d> &set_1, const std::vector<feature_2d> &set_2);

std::vector<feature_match> match_features_subset(const std::vector<feature_2d> &set_1,
                                                 const std::vector<feature_2d> &set_2,
                                                 const std::vector<size_t> &indices_1,
                                                 const std::vector<size_t> &indices_2);

std::vector<feature_match> match_features_local_guided(const std::vector<feature_2d> &set_1,
                                                       const std::vector<feature_2d> &set_2,
                                                       const Eigen::Matrix3d &homography, double search_radius_pixels,
                                                       const Eigen::Matrix3d *fundamental_matrix = nullptr,
                                                       double epipolar_threshold_pixels = 12.0);
} // namespace opencalibration
