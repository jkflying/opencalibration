#pragma once

#include <opencalibration/types/surface_model.hpp>

namespace opencalibration
{
bool toXYZ(const std::vector<surface_model> &surfaces, std::ostream &out);
}
