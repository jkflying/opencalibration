#pragma once

#include <string>
#include <eigen3/Eigen/Core>

namespace opencalibration
{

struct image_metadata
{
    size_t width_px = 0, height_px = 0;
    double focal_length_px = NAN;
    Eigen::Vector2d principal_point_px {NAN, NAN};

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


};
}
