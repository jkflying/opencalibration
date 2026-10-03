#pragma once

#include <eigen3/Eigen/Core>

#include <array>
#include <bitset>
#include <cstdint>

namespace opencalibration
{

struct Descriptor
{
    static constexpr int WORDS = 8;
    std::array<uint64_t, WORDS> words{};

    bool operator[](int i) const
    {
        return (words[i >> 6] >> (i & 63)) & 1;
    }
    void set(int i, bool value)
    {
        const uint64_t mask = uint64_t(1) << (i & 63);
        words[i >> 6] = value ? (words[i >> 6] | mask) : (words[i >> 6] & ~mask);
    }
    bool operator==(const Descriptor &other) const
    {
        return words == other.words;
    }
};

inline int hammingDistance(const Descriptor &a, const Descriptor &b)
{
    int distance = 0;
    for (int k = 0; k < Descriptor::WORDS; k++)
        distance += static_cast<int>(std::bitset<64>(a.words[k] ^ b.words[k]).count());
    return distance;
}

struct feature_2d
{
    static constexpr int DESCRIPTOR_BITS = 486;
    static_assert(DESCRIPTOR_BITS <= Descriptor::WORDS * 64);

    Eigen::Vector2d location = {NAN, NAN};
    float strength = 0;
    Descriptor descriptor;

    bool operator==(const feature_2d &other) const
    {
        return location == other.location && descriptor == other.descriptor && strength == other.strength;
    }
};

} // namespace opencalibration
