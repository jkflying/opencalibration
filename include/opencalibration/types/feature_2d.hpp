#pragma once

#include <bitset>
#include <eigen3/Eigen/Core>

namespace opencalibration
{

struct feature_2d
{
    static constexpr int DESCRIPTOR_BITS = 256;

    Eigen::Vector2d location = {NAN, NAN};
    float strength = 0;
    std::bitset<DESCRIPTOR_BITS> descriptor{};

    bool operator==(const feature_2d &other) const
    {
        return location == other.location && descriptor == other.descriptor && strength == other.strength;
    }
};

} // namespace opencalibration
