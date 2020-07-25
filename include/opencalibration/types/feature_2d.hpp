#pragma once

#include <bitset>
#include <Eigen/Core>

namespace opencalibration
{

struct feature_2d
{
    static constexpr int DESCRIPTOR_BITS = 256;

    Eigen::Vector2d location = {NAN, NAN};
    std::bitset<DESCRIPTOR_BITS> descriptor {};
    float strength = 0;
};

} // namespace opencalibration
