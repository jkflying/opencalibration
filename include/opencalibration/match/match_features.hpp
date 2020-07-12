#pragma once

#include <opencalibration/types/feature_2d.hpp>
#include <opencalibration/types/feature_match.hpp>

#include <vector>
#include <tuple>

namespace opencalibration
{
std::vector<feature_match> match_features(const std::vector<feature_2d>& set_1, const std::vector<feature_2d>& set_2);
}
