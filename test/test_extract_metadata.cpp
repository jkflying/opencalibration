#include <opencalibration/extract/extract_metadata.hpp>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>

using namespace opencalibration;
TEST(extract_metadata, gives_exif)
{
    // GIVEN: a path
    std::string path = TEST_DATA_DIR "P2530253.JPG";

    // WHEN: we extract the features
    image_metadata d = opencalibration::extract_metadata(path);

    // THEN: it should be these values for the file specified:
    EXPECT_EQ(d.width_px, 5344);
    EXPECT_EQ(d.height_px, 4016);
    EXPECT_NEAR(d.focal_length_px, 3553.28, 0.1);
    EXPECT_DOUBLE_EQ(d.latitude, -34.020866540722224);
}
