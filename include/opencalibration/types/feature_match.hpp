#pragma once

#include <stddef.h>

namespace opencalibration
{

struct feature_match
{
    size_t feature_index_1;
    size_t feature_index_2;
    double distance;

    bool operator==(const feature_match &other) const
    {
        return feature_index_1 == other.feature_index_1 && feature_index_2 == other.feature_index_2 &&
               distance == other.distance;
    }
};
} // namespace opencalibration
