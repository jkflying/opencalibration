#include <opencalibration/extract/extract_metadata.hpp>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>

using namespace opencalibration;
TEST(extract_metadata, gives_exif)
{
    // GIVEN: a path
    std::string path = TEST_DATA_DIR "IMG_1378_RGB.jpg";

    // WHEN: we extract the features
    image_metadata d = opencalibration::extract_metadata(path);

    // THEN: it should be these values for the file specified:
    EXPECT_EQ(d.camera_info.width_px, 4000);
    EXPECT_EQ(d.camera_info.height_px, 3000);
    EXPECT_NEAR(d.camera_info.focal_length_px, 2795.7, 1.0);
    EXPECT_DOUBLE_EQ(d.capture_info.latitude, 41.227004460009582);
    EXPECT_NEAR(d.capture_info.longitude, -81.702642, 1e-5);
}
