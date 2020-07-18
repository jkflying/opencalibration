#include <opencalibration/extract/extract_metadata.hpp>

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

using namespace opencalibration;
TEST(extract_metadata, gives_exif)
{
    // GIVEN: a path
    std::string path = TEST_DATA_DIR "P2530253.JPG";

    // WHEN: we extract the features
    image_metadata d = opencalibration::extract_metadata(path);

    std::cout << d.width_px << " " << d.height_px << " " << d.focal_length_px << " " << d.principal_point_px.transpose()
              << "\n"
              << d.latitude << " " << d.longitude << " " << d.altitude << " " << d.relativeAltitude << "\n"
              << d.rollDegree << " " << d.pitchDegree << " " << d.yawDegree << "\n"
              << d.accuracyXY << " " << d.accuracyZ << "\n"
              << d.datum << " " << d.timestamp << " " << d.datestamp << std::endl;
    // THEN: it should be these values for the file specified:
    EXPECT_EQ(d.width_px, 5344);
    EXPECT_EQ(d.height_px, 4016);
    EXPECT_NEAR(d.focal_length_px, 3553.28, 0.1);
    EXPECT_DOUBLE_EQ(d.latitude, -34.020866540722224);
}
