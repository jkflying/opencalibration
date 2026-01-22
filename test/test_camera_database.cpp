#include <opencalibration/extract/camera_database.hpp>
#include <opencalibration/extract/extract_image.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

class CameraDatabaseTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Database should be auto-loaded by extract_image or can be loaded explicitly
    }
};

TEST_F(CameraDatabaseTest, loads_bundled_database)
{
    // Database is loaded from CAMERA_DATABASE_PATH defined at compile time
    // After extract_image is called, it should be loaded
    auto img = extract_image(TEST_DATA_DIR "P2530253.JPG");
    ASSERT_TRUE(img.has_value());

    EXPECT_TRUE(CameraDatabase::instance().isLoaded());
}

TEST_F(CameraDatabaseTest, lookup_parrot_anafi)
{
    // First trigger database load
    auto img = extract_image(TEST_DATA_DIR "P2530253.JPG");
    ASSERT_TRUE(img.has_value());

    image_metadata::camera_info_t info;
    info.make = "Parrot";
    info.model = "Anafi";
    info.width_px = 5344;
    info.height_px = 4016;

    auto entry = CameraDatabase::instance().lookup(info);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->make, "Parrot");
    EXPECT_EQ(entry->model, "Anafi");
    EXPECT_EQ(entry->radial_distortion[0], -0.205);
}

TEST_F(CameraDatabaseTest, lookup_case_insensitive)
{
    // First trigger database load
    auto img = extract_image(TEST_DATA_DIR "P2530253.JPG");
    ASSERT_TRUE(img.has_value());

    image_metadata::camera_info_t info;
    info.make = "PARROT";
    info.model = "anafi";
    info.width_px = 5344;
    info.height_px = 4016;

    auto entry = CameraDatabase::instance().lookup(info);
    ASSERT_TRUE(entry.has_value());
}

TEST_F(CameraDatabaseTest, lookup_returns_nullopt_for_unknown)
{
    // First trigger database load
    auto img = extract_image(TEST_DATA_DIR "P2530253.JPG");
    ASSERT_TRUE(img.has_value());

    image_metadata::camera_info_t info;
    info.make = "Unknown";
    info.model = "Camera";
    info.width_px = 1000;
    info.height_px = 1000;

    auto entry = CameraDatabase::instance().lookup(info);
    EXPECT_FALSE(entry.has_value());
}

TEST_F(CameraDatabaseTest, extract_image_looks_up_database)
{
    // GIVEN: a path to a Parrot Anafi image
    std::string path = TEST_DATA_DIR "P2530253.JPG";

    // WHEN: we extract the image
    auto img = extract_image(path);

    // THEN: database should be loaded (distortion may be zero if no factory calibration)
    ASSERT_TRUE(img.has_value());
    EXPECT_TRUE(CameraDatabase::instance().isLoaded());

    // Verify the lookup works
    auto entry = CameraDatabase::instance().lookup(img->metadata.camera_info);
    EXPECT_TRUE(entry.has_value());
}

TEST_F(CameraDatabaseTest, apply_entry_sets_distortion)
{
    CameraDBEntry entry;
    entry.radial_distortion = Eigen::Vector3d(-0.02, 0.005, 0.0);
    entry.tangential_distortion = Eigen::Vector2d(0.001, -0.001);
    entry.principal_point_offset = Eigen::Vector2d(5.0, -3.0);
    entry.sensor_width_px = 5344;
    entry.sensor_height_px = 4016;

    image_metadata::camera_info_t camera_info;
    camera_info.width_px = 5344;
    camera_info.height_px = 4016;

    CameraModel model;
    model.pixels_cols = 5344;
    model.pixels_rows = 4016;
    model.principle_point = Eigen::Vector2d(2672, 2008);

    applyDatabaseEntry(entry, camera_info, model);

    EXPECT_EQ(model.radial_distortion, entry.radial_distortion);
    EXPECT_EQ(model.tangential_distortion, entry.tangential_distortion);
    EXPECT_NEAR(model.principle_point.x(), 2677.0, 0.01);
    EXPECT_NEAR(model.principle_point.y(), 2005.0, 0.01);
}

TEST_F(CameraDatabaseTest, apply_entry_scales_principal_point_for_different_resolution)
{
    CameraDBEntry entry;
    entry.radial_distortion = Eigen::Vector3d(-0.02, 0.005, 0.0);
    entry.tangential_distortion = Eigen::Vector2d(0.0, 0.0);
    entry.principal_point_offset = Eigen::Vector2d(10.0, 10.0);
    entry.sensor_width_px = 5344;
    entry.sensor_height_px = 4016;

    // Half resolution image
    image_metadata::camera_info_t camera_info;
    camera_info.width_px = 2672;
    camera_info.height_px = 2008;

    CameraModel model;
    model.pixels_cols = 2672;
    model.pixels_rows = 2008;
    model.principle_point = Eigen::Vector2d(1336, 1004);

    applyDatabaseEntry(entry, camera_info, model);

    // Principal point offset should be scaled by 0.5
    EXPECT_NEAR(model.principle_point.x(), 1341.0, 0.01);
    EXPECT_NEAR(model.principle_point.y(), 1009.0, 0.01);
}
