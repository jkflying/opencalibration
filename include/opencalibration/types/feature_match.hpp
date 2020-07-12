#pragma once

#include <stddef.h>

namespace opencalibration
{

struct feature_match
{
    size_t feature_index_1;
    size_t feature_index_2;
    double distance;
};
} // namespace opencalibration
