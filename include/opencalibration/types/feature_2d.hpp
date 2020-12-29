#pragma once

#include <bitset>
#include <eigen3/Eigen/Core>

namespace opencalibration
{

struct feature_2d
{
    static constexpr int DESCRIPTOR_BITS = 256;

    Eigen::Vector2d location = {NAN, NAN};
    std::bitset<DESCRIPTOR_BITS> descriptor{};
    float strength = 0;

    bool operator==(const feature_2d &other) const
    {
        return location == other.location && descriptor == other.descriptor && strength == other.strength;
    }
};

} // namespace opencalibration
