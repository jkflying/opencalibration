#pragma once

#include <opencalibration/types/feature_2d.hpp>
#include <opencalibration/types/feature_match.hpp>

#include <vector>

namespace opencalibration
{
std::vector<size_t> spatially_subsample_feature_indices(const std::vector<feature_2d> &features, double spacing_pixels,
                                                        size_t count = 0);

std::vector<feature_match> match_features_subset(const std::vector<feature_2d> &set_1,
                                                 const std::vector<feature_2d> &set_2,
                                                 const std::vector<size_t> &indices_1,
                                                 const std::vector<size_t> &indices_2);
} // namespace opencalibration
