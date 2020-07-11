#pragma once

#include <bitset>
#include <eigen3/Eigen/Core>

namespace opencalibration
{

struct feature_2d
{
    static constexpr int DESCRIPTOR_BITS = 128;

    Eigen::Vector2d location;
    std::bitset<DESCRIPTOR_BITS> descriptor;
    float strength;
};

} // namespace opencalibration
