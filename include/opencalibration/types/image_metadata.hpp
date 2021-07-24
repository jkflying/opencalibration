#pragma once

#include <eigen3/Eigen/Core>
#include <string>

namespace opencalibration
{

struct image_metadata
{
    struct camera_info_t
    {
        size_t width_px = 0, height_px = 0;
        double focal_length_px = NAN;
        Eigen::Vector2d principal_point_px{NAN, NAN};
        std::string make;
        std::string model;
        std::string serial_no;
        std::string lens_make;
        std::string lens_model;

        bool operator==(const camera_info_t &other) const
        {
            auto double_eq = [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); };
            // clang-format off
            return width_px == other.width_px &&
                height_px == other.height_px &&
                make == other.make &&
                model == other.model &&
                serial_no == other.serial_no &&
                lens_make == other.lens_make &&
                lens_model == other.lens_model &&
                double_eq(focal_length_px, other.focal_length_px);
            // clang-format on
        }

    } camera_info;

    struct capture_info_t
    {
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

        bool operator==(const capture_info_t &other) const
        {
            auto double_eq = [](double a, double b) { return a == b || (std::isnan(a) && std::isnan(b)); };
            // clang-format off
            return  double_eq(latitude, other.latitude) &&
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
    } capture_info;

    bool operator==(const image_metadata &other) const
    {
        return camera_info == other.camera_info && capture_info == other.capture_info;
    }
};
} // namespace opencalibration
