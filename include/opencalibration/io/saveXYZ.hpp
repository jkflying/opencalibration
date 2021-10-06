#pragma once

#include <opencalibration/types/surface_model.hpp>

namespace opencalibration
{
bool toXYZ(const std::vector<surface_model> &surfaces, std::ostream &out,
           const std::array<std::pair<int64_t, int64_t>, 3> &bounds = {});

std::array<std::pair<int64_t, int64_t>, 3> filterOutliers(const std::vector<surface_model> &surfaces);
} // namespace opencalibration
