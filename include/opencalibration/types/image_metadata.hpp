#pragma once

#include <eigen3/Eigen/Core>
#include <string>

namespace opencalibration
{

struct image_metadata
{
    size_t width_px = 0, height_px = 0;
    double focal_length_px = NAN;
    Eigen::Vector2d principal_point_px{NAN, NAN};

    double latitude = NAN;
    double longitude = NAN;
    double altitude = NAN;
    double relativeAltitude = NAN;

    double rollDegree = NAN;
    double pitchDegree = NAN;
    double yawDegree = NAN;

    double accuracyXY = NAN;
    double accuracyZ = NAN;

    std::string datum;
    std::string timestamp;
    std::string datestamp;

    bool operator==(const image_metadata &other) const
    {
        auto double_eq = [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); };
        // clang-format off
        return width_px == other.width_px &&
                height_px == other.height_px &&
                double_eq(focal_length_px , other.focal_length_px) &&
                double_eq(latitude, other.latitude) &&
                double_eq(longitude, other.longitude) &&
                double_eq(altitude, other.altitude) &&
                double_eq(relativeAltitude, other.relativeAltitude) &&
                double_eq(rollDegree, other.rollDegree) &&
                double_eq(pitchDegree, other.pitchDegree) &&
                double_eq(yawDegree, other.yawDegree) &&
                double_eq(accuracyXY, other.accuracyXY) &&
                double_eq(accuracyZ, other.accuracyZ) &&
                datum == other.datum &&
                timestamp == other.timestamp &&
                datestamp == other.datestamp;
        // clang-format on
    }
};
} // namespace opencalibration
