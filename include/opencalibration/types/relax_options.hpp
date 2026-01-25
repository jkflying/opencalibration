#pragma once

#include <bitset>
#include <initializer_list>

namespace opencalibration
{

enum class Option : int32_t
{
    ORIENTATION = 0,
    POSITION,

    GROUND_PLANE, // choose just one of these
    GROUND_MESH,
    POINTS_3D,

    FOCAL_LENGTH,
    PRINCIPAL_POINT,
    LENS_DISTORTIONS_RADIAL,

    LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION, // choose just one of these
    LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION,
    LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION,

    LENS_DISTORTIONS_TANGENTIAL,

    MINIMAL_MESH,
    ADAPTIVE_REFINE,

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

    bool hasAny(const RelaxOptionSet &other) const
    {
        return (enabled & other.enabled).count() > 0;
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

inline std::string toString(Option o)
{
    switch (o)
    {
    case Option::ORIENTATION:
        return "ORIENTATION";
    case Option::POSITION:
        return "POSITION";
    case Option::GROUND_PLANE:
        return "GROUND_PLANE";
    case Option::GROUND_MESH:
        return "GROUND_MESH";
    case Option::POINTS_3D:
        return "POINTS_3D";
    case Option::FOCAL_LENGTH:
        return "FOCAL_LENGTH";
    case Option::PRINCIPAL_POINT:
        return "PRINCIPAL_POINT";
    case Option::LENS_DISTORTIONS_RADIAL:
        return "LENS_DISTORTIONS_RADIAL";
    case Option::LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION:
        return "LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION";
    case Option::LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION:
        return "LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION";
    case Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION:
        return "LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION";
    case Option::LENS_DISTORTIONS_TANGENTIAL:
        return "LENS_DISTORTIONS_TANGENTIAL";
    case Option::MINIMAL_MESH:
        return "MINIMAL_MESH";
    case Option::ADAPTIVE_REFINE:
        return "ADAPTIVE_REFINE";
    case Option::__NUM_ENTRIES:
        return "__NUM_ENTRIES";
    }
    return "UNKNOWN";
}

inline std::string toString(RelaxOptionSet set)
{
    std::ostringstream oss;
    for (int32_t i = static_cast<int32_t>(Option::ORIENTATION); i < static_cast<int32_t>(Option::__NUM_ENTRIES); i++)
    {
        if (set.hasAll({static_cast<Option>(i)}))
        {
            if (oss.tellp() > 0)
                oss << ", ";
            oss << toString(static_cast<Option>(i));
        }
    }

    return oss.str();
}

} // namespace opencalibration
