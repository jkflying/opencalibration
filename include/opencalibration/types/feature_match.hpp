#pragma once

#include <eigen3/Eigen/Core>

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

// contains pre-looked-up pixel locations for quickly building cost functions
struct feature_match_denormalized
{
    Eigen::Vector2d pixel_1{NAN, NAN}, pixel_2{NAN, NAN};
    size_t feature_index_1, feature_index_2, match_index;

    bool operator==(const feature_match_denormalized &other) const
    {
        return ((pixel_1.array().isNaN().all() && other.pixel_1.array().isNaN().all()) || pixel_1 == other.pixel_1) &&
               ((pixel_2.array().isNaN().all() && other.pixel_2.array().isNaN().all()) || pixel_2 == other.pixel_2) &&
               feature_index_1 == other.feature_index_1 && feature_index_2 == other.feature_index_2 &&
               match_index == other.match_index;
    }
};
} // namespace opencalibration
