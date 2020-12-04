#include <opencalibration/extract/extract_metadata.hpp>

#include "TinyEXIF.h"

#include <fstream>
#include <iostream>
#include <vector>

namespace opencalibration
{

image_metadata extract_metadata(const std::string &path)
{

    std::ifstream file(path, std::ifstream::in | std::ifstream::binary);
    file.seekg(0, std::ios::end);
    std::streampos length = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(length);
    file.read((char *)data.data(), length);

    // parse image EXIF and XMP metadata
    TinyEXIF::EXIFInfo imageEXIF(data.data(), length);

    image_metadata res;
    if (!imageEXIF.Fields)
    {
        return res;
    }

    res.timestamp = imageEXIF.DateTimeOriginal.size() > 0 ? imageEXIF.DateTimeOriginal + imageEXIF.SubSecTimeOriginal
                                                          : imageEXIF.DateTime;

    if (imageEXIF.GeoLocation.hasLatLon())
    {
        res.latitude = imageEXIF.GeoLocation.Latitude;
        res.longitude = imageEXIF.GeoLocation.Longitude;
        if (imageEXIF.GeoLocation.AccuracyXY > 0)
        {
            res.accuracyXY = imageEXIF.GeoLocation.AccuracyXY;
        }

        res.datum = imageEXIF.GeoLocation.GPSMapDatum;
        if (imageEXIF.GeoLocation.GPSDateStamp.size() > 0)
        {
            res.timestamp = imageEXIF.GeoLocation.GPSDateStamp + " " + imageEXIF.GeoLocation.GPSTimeStamp;
        }
    }

    if (imageEXIF.GeoLocation.hasAltitude())
    {
        res.altitude = imageEXIF.GeoLocation.Altitude;
        if (imageEXIF.GeoLocation.AccuracyZ > 0)
        {
            res.accuracyZ = imageEXIF.GeoLocation.AccuracyZ;
        }
    }

    if (imageEXIF.GeoLocation.hasRelativeAltitude())
    {
        res.altitude = imageEXIF.GeoLocation.RelativeAltitude;
        res.accuracyZ = imageEXIF.GeoLocation.AccuracyZ;
    }

    if (imageEXIF.GeoLocation.hasOrientation())
    {
        res.rollDegree = imageEXIF.GeoLocation.RollDegree;
        res.pitchDegree = imageEXIF.GeoLocation.PitchDegree;
        res.yawDegree = imageEXIF.GeoLocation.YawDegree;
    }

    res.width_px = imageEXIF.ImageWidth;
    res.height_px = imageEXIF.ImageHeight;

    if (imageEXIF.Calibration.FocalLength > 1)
    {
        res.focal_length_px = imageEXIF.Calibration.FocalLength;
    }
    else if (imageEXIF.LensInfo.FocalLengthIn35mm > 0)
    {
        /*
         * From wikipedia:
         * Converted focal length into 35mm camera = (Diagonal distance of image area in the 35mm camera (43.27mm) /
         * Diagonal distance of image area on the image sensor of the DSC) × focal length of the lens of the DSC.
         *
         * eq35 = 43.27 / diag_pixels * focal_lens
         * focal_lens = eq35 / 43.27 * diag_pixels
         */

        res.focal_length_px = imageEXIF.LensInfo.FocalLengthIn35mm / 43.27 * std::hypot(res.width_px, res.height_px);
    }
    else if (imageEXIF.FocalLength > 0 && imageEXIF.LensInfo.FocalPlaneXResolution > 0)
    {
        /*
         * Use sensor and lens physical size to determine focal length in pixels
         */
        double scale = 25.4;
        if (imageEXIF.LensInfo.FocalPlaneResolutionUnit == 3)
        {
            scale = 10;
        }
        double pixel_size_mm = scale / imageEXIF.LensInfo.FocalPlaneXResolution;
        res.focal_length_px = imageEXIF.FocalLength / pixel_size_mm;
    }
    if (imageEXIF.Calibration.OpticalCenterX > 0 && imageEXIF.Calibration.OpticalCenterY > 0)
    {
        res.principal_point_px =
            Eigen::Vector2d(imageEXIF.Calibration.OpticalCenterY, imageEXIF.Calibration.OpticalCenterX);
    }

    return res;
}

} // namespace opencalibration
