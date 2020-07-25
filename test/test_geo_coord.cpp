#include <opencalibration/geo_coord/geo_coord.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

TEST(geo_coord, instantiates)
{
    GeoCoord g;
}

TEST(geo_coord, intializes)
{
    // GIVEN: an empty GeoCoord
    GeoCoord g;

    // THEN: it shouldn't be initialized
    EXPECT_FALSE(g.isInitialized());

    // WHEN: it gets an origin set
    g.setOrigin(-34, 18.23);

    // THEN: it should be initialized
    EXPECT_TRUE(g.isInitialized());

    // AND: the origin should translate to zero
    EXPECT_EQ(g.toLocalCS(-34, 18.23, 0), Eigen::Vector3d::Zero().eval()) << g.toLocalCS(-34, 18.23, 0).transpose();

    std::string WKT = "PROJCS[\"Custom Transverse Mercator\",\n"
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
                      "    PARAMETER[\"latitude_of_origin\",-34],\n"
                      "    PARAMETER[\"central_meridian\",18.23],\n"
                      "    PARAMETER[\"scale_factor\",1],\n"
                      "    PARAMETER[\"false_easting\",0],\n"
                      "    PARAMETER[\"false_northing\",0],\n"
                      "    UNIT[\"metre\",1,\n"
                      "        AUTHORITY[\"EPSG\",\"9001\"]],\n"
                      "    AXIS[\"Easting\",EAST],\n"
                      "    AXIS[\"Northing\",NORTH]]";

    // AND: the WKT should match (formatting might change on GDAL versions?)
    EXPECT_STREQ(g.getWKT().c_str(), WKT.c_str());
}
