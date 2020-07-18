#pragma once

#include <opencalibration/types/image.hpp>

#include <string>

namespace opencalibration
{

    image_metadata extract_metadata(const std::string& path);

}
