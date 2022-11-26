#pragma once

#include <opencalibration/types/image.hpp>

#include <optional>
#include <vector>

namespace opencalibration
{

std::optional<image> extract_image(const std::string &path);
}
