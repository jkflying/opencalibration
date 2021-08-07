#pragma once

#include <bitset>
#include <initializer_list>

namespace opencalibration
{

enum class Option : int32_t
{
    ORIENTATION = 0,
    POSITION,
    GROUND_PLANE,
    POINTS_3D,
    FOCAL_LENGTH,
    LENS_DISTORTIONS_RADIAL,

    LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION, // choose just one of these
    LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION,
    LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION,

    LENS_DISTORTIONS_TANGENTIAL,

    __NUM_ENTRIES
};

class RelaxOptionSet
{
  public:
    RelaxOptionSet()
    {
    }

    RelaxOptionSet(std::initializer_list<Option> list)
    {
        for (Option o : list)
        {
            set(o, true);
        }
    }

    bool get(Option o) const
    {
        return enabled[static_cast<int32_t>(o)];
    }

    void set(Option o, bool value)
    {
        enabled[static_cast<int32_t>(o)] = value;
    }

    bool hasAll(const RelaxOptionSet &other) const
    {
        return (other.enabled | enabled) == enabled;
    }

    bool operator==(const RelaxOptionSet &other) const
    {
        return other.enabled == enabled;
    }

    int32_t count() const
    {
        return static_cast<int32_t>(enabled.count());
    }

    int32_t excess(const RelaxOptionSet &other) const
    {
        return count() - other.count();
    }

  private:
    std::bitset<static_cast<int32_t>(Option::__NUM_ENTRIES)> enabled;
};

} // namespace opencalibration
