#include <opencalibration/extract/camera_database.hpp>
#include <opencalibration/extract/extract_image.hpp>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

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
    EXPECT_NEAR(entry->radial_distortion[0], -0.03227143641412748, 1e-12);
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

TEST_F(CameraDatabaseTest, update_database_from_graph_creates_new_file)
{
    // GIVEN: a graph with one camera model
    MeasurementGraph graph;
    image img;
    img.model = std::make_shared<CameraModel>();
    img.model->id = 1;
    img.model->pixels_cols = 4000;
    img.model->pixels_rows = 3000;
    img.model->principle_point = Eigen::Vector2d(2005.0, 1497.0);
    img.model->radial_distortion = Eigen::Vector3d(-0.1, 0.02, -0.001);
    img.model->tangential_distortion = Eigen::Vector2d(0.0005, -0.0003);
    img.metadata.camera_info.make = "TestMake";
    img.metadata.camera_info.model = "TestModel";
    img.metadata.camera_info.lens_model = "TestLens";
    graph.addNode(std::move(img));

    std::string output_path = TEST_DATA_OUTPUT_DIR "test_update_db.json";

    std::remove(output_path.c_str());

    // WHEN: we call updateDatabaseFromGraph
    bool result = updateDatabaseFromGraph(graph, output_path);

    // THEN: it succeeds and the file contains our entry
    ASSERT_TRUE(result);

    std::ifstream file(output_path);
    ASSERT_TRUE(file.is_open());
    std::stringstream buf;
    buf << file.rdbuf();
    std::string json = buf.str();

    EXPECT_NE(json.find("TestMake"), std::string::npos);
    EXPECT_NE(json.find("TestModel"), std::string::npos);
    EXPECT_NE(json.find("TestLens"), std::string::npos);
    EXPECT_NE(json.find("-0.1"), std::string::npos);
}

TEST_F(CameraDatabaseTest, update_database_from_graph_merges_existing)
{
    // GIVEN: an existing database file with one entry
    std::string output_path = TEST_DATA_OUTPUT_DIR "test_update_db_merge.json";
    {
        std::ofstream out(output_path);
        out << R"({"version":1,"cameras":[{"make":"TestMake","model":"TestModel","lens_model":"TestLens",)"
            << R"("sensor_width_px":4000,"sensor_height_px":3000,)"
            << R"("radial_distortion":[-0.05,0.01,0.0],)"
            << R"("tangential_distortion":[0.0,0.0],)"
            << R"("principal_point_offset":[0.0,0.0],)"
            << R"("notes":"original"}]})";
    }

    // AND: a graph with the same camera but updated parameters
    MeasurementGraph graph;
    image img;
    img.model = std::make_shared<CameraModel>();
    img.model->id = 1;
    img.model->pixels_cols = 4000;
    img.model->pixels_rows = 3000;
    img.model->principle_point = Eigen::Vector2d(2005.0, 1497.0);
    img.model->radial_distortion = Eigen::Vector3d(-0.2, 0.03, -0.002);
    img.model->tangential_distortion = Eigen::Vector2d(0.001, -0.001);
    img.metadata.camera_info.make = "TestMake";
    img.metadata.camera_info.model = "TestModel";
    img.metadata.camera_info.lens_model = "TestLens";
    graph.addNode(std::move(img));

    // WHEN: we call updateDatabaseFromGraph
    bool result = updateDatabaseFromGraph(graph, output_path);

    // THEN: it succeeds, has exactly one entry (merged, not duplicated), and notes are preserved
    ASSERT_TRUE(result);

    std::ifstream file(output_path);
    ASSERT_TRUE(file.is_open());
    std::stringstream buf;
    buf << file.rdbuf();
    std::string json = buf.str();

    EXPECT_NE(json.find("-0.2"), std::string::npos);
    EXPECT_NE(json.find("original"), std::string::npos);
    // Not duplicated
    size_t first = json.find("TestMake");
    size_t second = json.find("TestMake", first + 1);
    EXPECT_EQ(second, std::string::npos);
}

TEST_F(CameraDatabaseTest, extract_entry_includes_focal_length_pixels)
{
    // GIVEN: an image with known focal length, verified via updateDatabaseFromGraph
    MeasurementGraph graph;
    image img;
    img.model = std::make_shared<CameraModel>();
    img.model->id = 1;
    img.model->pixels_cols = 5000;
    img.model->pixels_rows = 3750;
    img.model->focal_length_pixels = 5500.0;
    img.model->principle_point = Eigen::Vector2d(2500, 1875);
    img.model->radial_distortion = Eigen::Vector3d(0, 0, 0);
    img.model->tangential_distortion = Eigen::Vector2d(0, 0);
    img.metadata.camera_info.make = "TestMake";
    img.metadata.camera_info.model = "TestModel";
    img.metadata.camera_info.lens_model = "TestLens";
    graph.addNode(std::move(img));

    std::string output_path = TEST_DATA_OUTPUT_DIR "test_extract_focal.json";
    std::remove(output_path.c_str());

    // WHEN: we save to database (this uses extractEntry internally)
    ASSERT_TRUE(updateDatabaseFromGraph(graph, output_path));

    // THEN: focal_length_pixels should be in the JSON and correctly preserved
    std::ifstream file(output_path);
    ASSERT_TRUE(file.is_open());
    std::stringstream buf;
    buf << file.rdbuf();
    std::string json = buf.str();

    EXPECT_NE(json.find("focal_length_pixels"), std::string::npos);
    EXPECT_NE(json.find("5500"), std::string::npos);
}

TEST_F(CameraDatabaseTest, extract_entry_handles_missing_focal_length)
{
    // GIVEN: an image without focal length (NaN), verified via updateDatabaseFromGraph
    MeasurementGraph graph;
    image img;
    img.model = std::make_shared<CameraModel>();
    img.model->id = 1;
    img.model->pixels_cols = 5000;
    img.model->pixels_rows = 3750;
    img.model->focal_length_pixels = NAN;
    img.model->principle_point = Eigen::Vector2d(2500, 1875);
    img.model->radial_distortion = Eigen::Vector3d(0, 0, 0);
    img.model->tangential_distortion = Eigen::Vector2d(0, 0);
    img.metadata.camera_info.make = "TestMake";
    img.metadata.camera_info.model = "TestModel";
    img.metadata.camera_info.lens_model = "TestLens";
    graph.addNode(std::move(img));

    std::string output_path = TEST_DATA_OUTPUT_DIR "test_missing_focal.json";
    std::remove(output_path.c_str());

    // WHEN: we save to database
    ASSERT_TRUE(updateDatabaseFromGraph(graph, output_path));

    // THEN: focal_length_scale should NOT be in JSON (remains NaN)
    std::ifstream file(output_path);
    ASSERT_TRUE(file.is_open());
    std::stringstream buf;
    buf << file.rdbuf();
    std::string json = buf.str();

    EXPECT_EQ(json.find("focal_length_scale"), std::string::npos);
}

TEST_F(CameraDatabaseTest, serialize_focal_length_pixels)
{
    // GIVEN: database entries with focal_length_pixels
    MeasurementGraph graph;
    image img;
    img.model = std::make_shared<CameraModel>();
    img.model->id = 1;
    img.model->pixels_cols = 4000;
    img.model->pixels_rows = 3000;
    img.model->focal_length_pixels = 4400.0;
    img.model->principle_point = Eigen::Vector2d(2005.0, 1497.0);
    img.model->radial_distortion = Eigen::Vector3d(-0.1, 0.02, -0.001);
    img.model->tangential_distortion = Eigen::Vector2d(0.0005, -0.0003);
    img.metadata.camera_info.make = "TestMake";
    img.metadata.camera_info.model = "TestModel";
    img.metadata.camera_info.lens_model = "TestLens";
    graph.addNode(std::move(img));

    std::string output_path = TEST_DATA_OUTPUT_DIR "test_focal_serialize.json";
    std::remove(output_path.c_str());

    // WHEN: we save and reload the database
    ASSERT_TRUE(updateDatabaseFromGraph(graph, output_path));

    std::ifstream file(output_path);
    ASSERT_TRUE(file.is_open());
    std::stringstream buf;
    buf << file.rdbuf();
    std::string json = buf.str();

    // THEN: focal_length_pixels should be in the JSON
    EXPECT_NE(json.find("focal_length_pixels"), std::string::npos);
    EXPECT_NE(json.find("4400"), std::string::npos);
}

TEST_F(CameraDatabaseTest, backward_compatible_with_missing_focal_pixels)
{
    // GIVEN: an old database without focal_length_pixels
    // This tests that loadDatabaseEntries() gracefully handles missing focal_length_pixels
    std::string output_path = TEST_DATA_OUTPUT_DIR "test_backward_compat.json";
    {
        std::ofstream out(output_path);
        out << R"({"version":1,"cameras":[{"make":"OldMake","model":"OldModel","lens_model":"OldLens",)"
            << R"("sensor_width_px":4000,"sensor_height_px":3000,)"
            << R"("radial_distortion":[-0.05,0.01,0.0],)"
            << R"("tangential_distortion":[0.0,0.0],)"
            << R"("principal_point_offset":[0.0,0.0]}]})";
    }

    // WHEN: we load and save a database from the old format
    MeasurementGraph graph;
    image img;
    img.model = std::make_shared<CameraModel>();
    img.model->id = 1;
    img.model->pixels_cols = 4000;
    img.model->pixels_rows = 3000;
    img.model->principle_point = Eigen::Vector2d(2000, 1500);
    img.model->radial_distortion = Eigen::Vector3d(0, 0, 0);
    img.model->tangential_distortion = Eigen::Vector2d(0, 0);
    img.model->focal_length_pixels = NAN;
    img.metadata.camera_info.make = "OldMake";
    img.metadata.camera_info.model = "OldModel";
    img.metadata.camera_info.lens_model = "OldLens";
    graph.addNode(std::move(img));

    // Update database preserving old entry
    bool result = updateDatabaseFromGraph(graph, output_path);

    // THEN: it should succeed and preserve entries without focal_length_pixels
    ASSERT_TRUE(result);

    std::ifstream file(output_path);
    ASSERT_TRUE(file.is_open());
    std::stringstream buf;
    buf << file.rdbuf();
    std::string json = buf.str();

    EXPECT_NE(json.find("OldMake"), std::string::npos);
    EXPECT_NE(json.find("OldModel"), std::string::npos);
}

TEST_F(CameraDatabaseTest, apply_focal_length_when_exif_missing)
{
    // GIVEN: a database entry with focal_length_pixels and a model without EXIF focal length
    CameraDBEntry entry;
    entry.radial_distortion = Eigen::Vector3d(0, 0, 0);
    entry.tangential_distortion = Eigen::Vector2d(0, 0);
    entry.principal_point_offset = Eigen::Vector2d(0, 0);
    entry.sensor_width_px = 5000;
    entry.sensor_height_px = 3750;
    entry.focal_length_pixels = 5500.0;

    image_metadata::camera_info_t camera_info;
    camera_info.width_px = 5000;
    camera_info.height_px = 3750;

    CameraModel model;
    model.pixels_cols = 5000;
    model.pixels_rows = 3750;
    model.principle_point = Eigen::Vector2d(2500, 1875);
    model.focal_length_pixels = NAN;

    // WHEN: we apply the database entry
    applyDatabaseEntry(entry, camera_info, model);

    // THEN: focal_length_pixels should be set from database
    EXPECT_FALSE(std::isnan(model.focal_length_pixels));
    EXPECT_NEAR(model.focal_length_pixels, 5500.0, 0.01);
}

TEST_F(CameraDatabaseTest, preserve_exif_focal_length_over_database)
{
    // GIVEN: a database entry with focal_length_pixels and a model with EXIF focal length
    CameraDBEntry entry;
    entry.radial_distortion = Eigen::Vector3d(0, 0, 0);
    entry.tangential_distortion = Eigen::Vector2d(0, 0);
    entry.principal_point_offset = Eigen::Vector2d(0, 0);
    entry.sensor_width_px = 5000;
    entry.sensor_height_px = 3750;
    entry.focal_length_pixels = 5500.0;

    image_metadata::camera_info_t camera_info;
    camera_info.width_px = 5000;
    camera_info.height_px = 3750;

    CameraModel model;
    model.pixels_cols = 5000;
    model.pixels_rows = 3750;
    model.principle_point = Eigen::Vector2d(2500, 1875);
    model.focal_length_pixels = 5000.0; // EXIF focal length

    // WHEN: we apply the database entry
    applyDatabaseEntry(entry, camera_info, model);

    // THEN: EXIF focal_length_pixels should be preserved (not overwritten)
    EXPECT_NEAR(model.focal_length_pixels, 5000.0, 0.01);
}

TEST_F(CameraDatabaseTest, apply_focal_length_at_original_resolution)
{
    // GIVEN: a database entry with focal_length_pixels from full-res camera
    CameraDBEntry entry;
    entry.radial_distortion = Eigen::Vector3d(0, 0, 0);
    entry.tangential_distortion = Eigen::Vector2d(0, 0);
    entry.principal_point_offset = Eigen::Vector2d(0, 0);
    entry.sensor_width_px = 5000;
    entry.sensor_height_px = 3750;
    entry.focal_length_pixels = 5000.0;

    image_metadata::camera_info_t camera_info;
    camera_info.width_px = 5000;
    camera_info.height_px = 3750;

    CameraModel model;
    model.pixels_cols = 5000;
    model.pixels_rows = 3750;
    model.principle_point = Eigen::Vector2d(2500, 1875);
    model.focal_length_pixels = NAN;

    // WHEN: we apply the database entry
    applyDatabaseEntry(entry, camera_info, model);

    // THEN: focal_length_pixels should be applied as-is (from database for same resolution)
    EXPECT_FALSE(std::isnan(model.focal_length_pixels));
    EXPECT_NEAR(model.focal_length_pixels, 5000.0, 0.01);
}

TEST_F(CameraDatabaseTest, update_database_preserves_focal_length_pixels)
{
    // GIVEN: an existing database with focal_length_pixels
    std::string output_path = TEST_DATA_OUTPUT_DIR "test_update_db_focal.json";
    {
        std::ofstream out(output_path);
        out << R"({"version":1,"cameras":[{"make":"TestMake","model":"TestModel","lens_model":"TestLens",)"
            << R"("sensor_width_px":4000,"sensor_height_px":3000,)"
            << R"("radial_distortion":[-0.05,0.01,0.0],)"
            << R"("tangential_distortion":[0.0,0.0],)"
            << R"("principal_point_offset":[0.0,0.0],)"
            << R"("focal_length_pixels":4200.0}]})";
    }

    // AND: a graph with updated camera parameters and updated focal length
    MeasurementGraph graph;
    image img;
    img.model = std::make_shared<CameraModel>();
    img.model->id = 1;
    img.model->pixels_cols = 4000;
    img.model->pixels_rows = 3000;
    img.model->focal_length_pixels = 4400.0;
    img.model->principle_point = Eigen::Vector2d(2005.0, 1497.0);
    img.model->radial_distortion = Eigen::Vector3d(-0.2, 0.03, -0.002);
    img.model->tangential_distortion = Eigen::Vector2d(0.001, -0.001);
    img.metadata.camera_info.make = "TestMake";
    img.metadata.camera_info.model = "TestModel";
    img.metadata.camera_info.lens_model = "TestLens";
    graph.addNode(std::move(img));

    // WHEN: we update the database
    bool result = updateDatabaseFromGraph(graph, output_path);

    // THEN: it succeeds and the new focal_length_pixels is preserved
    ASSERT_TRUE(result);

    std::ifstream file(output_path);
    ASSERT_TRUE(file.is_open());
    std::stringstream buf;
    buf << file.rdbuf();
    std::string json = buf.str();

    EXPECT_NE(json.find("focal_length_pixels"), std::string::npos);
    EXPECT_NE(json.find("4400"), std::string::npos);
}
