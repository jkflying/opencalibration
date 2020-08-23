#include <opencalibration/geo_coord/geo_coord.hpp>

#include <gdal/ogr_spatialref.h>
#include <spdlog/spdlog.h>

#include <sstream>

namespace opencalibration
{
GeoCoord::~GeoCoord()
{
}
GeoCoord::GeoCoord()
{
}

bool GeoCoord::isInitialized()
{
    return _transform != nullptr;
}

bool GeoCoord::setOrigin(double latitude, double longitude)
{
    _transform.reset();

    OGRSpatialReference source, dest;
    const OGRErr errS = source.SetGeogCS("Standard WGS84", "World Geodetic System 1984", "WGS84 Spheroid",
                                         SRS_WGS84_SEMIMAJOR, SRS_WGS84_INVFLATTENING, "Greenwich", 0.0);

    std::ostringstream dest_wkt_stream;
    dest_wkt_stream << "PROJCS[\"Custom Transverse Mercator\",\n"
                       "    GEOGCS[\"WGS 84\",\n"
                       "        DATUM[\"WGS_1984\",\n"
                       "            SPHEROID[\"WGS 84\",6378137,298.257223563,\n"
                       "                AUTHORITY[\"EPSG\",\"7030\"]],\n"
                       "            AUTHORITY[\"EPSG\",\"6326\"]],\n"
                       "        PRIMEM[\"Greenwich\",0,\n"
                       "            AUTHORITY[\"EPSG\",\"8901\"]],\n"
                       "        UNIT[\"degree\",0.0174532925199433,\n"
                       "            AUTHORITY[\"EPSG\",\"9122\"]],\n"
                       "        AUTHORITY[\"EPSG\",\"4326\"]],\n"
                       "    PROJECTION[\"Transverse_Mercator\"],\n"
                       "    PARAMETER[\"latitude_of_origin\","
                    << latitude
                    << "],\n"
                       "    PARAMETER[\"central_meridian\","
                    << longitude
                    << "],\n"
                       "    PARAMETER[\"scale_factor\",1],\n"
                       "    PARAMETER[\"false_easting\",0],\n"
                       "    PARAMETER[\"false_northing\",0],\n"
                       "    UNIT[\"metre\",1,\n"
                       "        AUTHORITY[\"EPSG\",\"9001\"]],\n"
                       "    AXIS[\"Easting\",EAST],\n"
                       "    AXIS[\"Northing\",NORTH]]";

    // necessary for GDAL 2.4 *and* 3.0+ compatibility
#if (GDAL_VERSION_MAJOR >= 3)
    source.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    dest.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

    std::string dest_wkt_str = dest_wkt_stream.str();
    spdlog::info("Dest WKT:  \n{}", dest_wkt_str);

    const OGRErr errD = dest.importFromWkt(dest_wkt_str.c_str());
    if (errS == OGRERR_NONE && errD == OGRERR_NONE)
    {
        _transform.reset(OGRCreateCoordinateTransformation(&source, &dest));
    }

    if (_transform == nullptr)
    {
        spdlog::error("Unabled to generate transform: errS {}  errD {}", errS, errD);
    }

    return _transform != nullptr;
}

Eigen::Vector3d GeoCoord::toLocalCS(double latitude, double longitude, double altitude)
{
    Eigen::Vector3d res{longitude, latitude, altitude};
    bool success = false;
    if (_transform != nullptr)
    {
        success = TRUE == _transform->Transform(1, &res[0], &res[1], &res[2]);
    }
    if (success)
    {
        spdlog::debug("transformed global coordinate {},{},{} to local {}, {}, {}", latitude, longitude, altitude,
                      res.x(), res.y(), res.z());
    }
    else
    {
        spdlog::warn("unable to transform global coord {},{},{} to local {},{},{}", latitude, longitude, altitude,
                     res.x(), res.y(), res.z());
        res = Eigen::Vector3d(NAN, NAN, NAN);
    }
    return res;
}

std::string GeoCoord::getWKT()
{
    std::string res = "UNKNOWN";

    if (_transform != nullptr)
    {
        char *str;
        OGRErr err = _transform->GetTargetCS()->exportToPrettyWkt(&str);
        if (err == OGRERR_NONE)
        {
            res = std::string(str);
        }
        CPLFree(str);
    }

    return res;
}

} // namespace opencalibration
