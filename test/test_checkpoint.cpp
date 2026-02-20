#include <opencalibration/io/checkpoint.hpp>
#include <opencalibration/pipeline/pipeline.hpp>

#include <gtest/gtest.h>

#include <filesystem>

using namespace opencalibration;

class CheckpointTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        test_checkpoint_dir = std::string(TEST_DATA_OUTPUT_DIR) + "test_checkpoint";
        std::filesystem::remove_all(test_checkpoint_dir);
    }

    void TearDown() override
    {
        std::filesystem::remove_all(test_checkpoint_dir);
    }

    std::string test_checkpoint_dir;
};

TEST_F(CheckpointTest, save_and_load_empty)
{
    CheckpointData data;
    data.origin_latitude = 47.123456;
    data.origin_longitude = -122.654321;
    data.state = PipelineState::INITIAL_GLOBAL_RELAX;
    data.state_run_count = 5;

    ASSERT_TRUE(saveCheckpoint(data, test_checkpoint_dir));
    ASSERT_TRUE(validateCheckpoint(test_checkpoint_dir));

    CheckpointData loaded;
    ASSERT_TRUE(loadCheckpoint(test_checkpoint_dir, loaded));

    EXPECT_DOUBLE_EQ(data.origin_latitude, loaded.origin_latitude);
    EXPECT_DOUBLE_EQ(data.origin_longitude, loaded.origin_longitude);
    EXPECT_EQ(data.state, loaded.state);
    EXPECT_EQ(data.state_run_count, loaded.state_run_count);
    EXPECT_EQ(data.graph.size_nodes(), loaded.graph.size_nodes());
    EXPECT_EQ(data.surfaces.size(), loaded.surfaces.size());
}

TEST_F(CheckpointTest, save_and_load_with_surfaces)
{
    CheckpointData data;
    data.origin_latitude = 48.0;
    data.origin_longitude = -120.0;
    data.state = PipelineState::FINAL_GLOBAL_RELAX;
    data.state_run_count = 2;

    // Add some surfaces with point clouds
    surface_model surface1;
    point_cloud cloud1;
    cloud1.push_back(Eigen::Vector3d(1.0, 2.0, 3.0));
    cloud1.push_back(Eigen::Vector3d(4.0, 5.0, 6.0));
    surface1.cloud.push_back(cloud1);

    point_cloud cloud2;
    cloud2.push_back(Eigen::Vector3d(7.0, 8.0, 9.0));
    surface1.cloud.push_back(cloud2);

    data.surfaces.push_back(surface1);

    surface_model surface2;
    point_cloud cloud3;
    cloud3.push_back(Eigen::Vector3d(10.0, 11.0, 12.0));
    surface2.cloud.push_back(cloud3);
    data.surfaces.push_back(surface2);

    ASSERT_TRUE(saveCheckpoint(data, test_checkpoint_dir));
    ASSERT_TRUE(validateCheckpoint(test_checkpoint_dir));

    CheckpointData loaded;
    ASSERT_TRUE(loadCheckpoint(test_checkpoint_dir, loaded));

    EXPECT_EQ(data.surfaces.size(), loaded.surfaces.size());
    ASSERT_EQ(2u, loaded.surfaces.size());

    EXPECT_EQ(2u, loaded.surfaces[0].cloud.size());
    ASSERT_EQ(2u, loaded.surfaces[0].cloud[0].size());
    EXPECT_DOUBLE_EQ(1.0, loaded.surfaces[0].cloud[0][0].x());
    EXPECT_DOUBLE_EQ(2.0, loaded.surfaces[0].cloud[0][0].y());
    EXPECT_DOUBLE_EQ(3.0, loaded.surfaces[0].cloud[0][0].z());

    EXPECT_EQ(1u, loaded.surfaces[0].cloud[1].size());
    EXPECT_DOUBLE_EQ(7.0, loaded.surfaces[0].cloud[1][0].x());

    EXPECT_EQ(1u, loaded.surfaces[1].cloud.size());
    EXPECT_EQ(1u, loaded.surfaces[1].cloud[0].size());
    EXPECT_DOUBLE_EQ(10.0, loaded.surfaces[1].cloud[0][0].x());
}

TEST_F(CheckpointTest, pipeline_save_and_load)
{
    Pipeline p1(1);
    p1.set_generate_thumbnails(false);

    // Note: This test doesn't actually run the pipeline, just tests the checkpoint infrastructure
    ASSERT_TRUE(p1.saveCheckpoint(test_checkpoint_dir));

    Pipeline p2(1);
    ASSERT_TRUE(p2.loadCheckpoint(test_checkpoint_dir));

    EXPECT_EQ(p1.getState(), p2.getState());
}

TEST_F(CheckpointTest, validate_nonexistent)
{
    EXPECT_FALSE(validateCheckpoint("/nonexistent/path/to/checkpoint"));
}

TEST_F(CheckpointTest, load_nonexistent)
{
    CheckpointData data;
    EXPECT_FALSE(loadCheckpoint("/nonexistent/path/to/checkpoint", data));
}

TEST_F(CheckpointTest, fromString_toString_roundtrip)
{
    std::vector<PipelineState> states = {
        PipelineState::INITIAL_PROCESSING, PipelineState::INITIAL_GLOBAL_RELAX, PipelineState::CAMERA_PARAMETER_RELAX,
        PipelineState::FINAL_GLOBAL_RELAX, PipelineState::GENERATE_THUMBNAIL,   PipelineState::GENERATE_LAYERS,
        PipelineState::COLOR_BALANCE,      PipelineState::BLEND_LAYERS,         PipelineState::COMPLETE};

    std::vector<std::string> state_strings = {"INITIAL_PROCESSING", "INITIAL_GLOBAL_RELAX", "CAMERA_PARAMETER_RELAX",
                                              "FINAL_GLOBAL_RELAX", "GENERATE_THUMBNAIL",   "GENERATE_LAYERS",
                                              "COLOR_BALANCE",      "BLEND_LAYERS",         "COMPLETE"};

    for (size_t i = 0; i < states.size(); i++)
    {
        auto parsed = Pipeline::fromString(state_strings[i]);
        ASSERT_TRUE(parsed.has_value()) << "Failed to parse: " << state_strings[i];
        EXPECT_EQ(states[i], *parsed);
    }
}

TEST_F(CheckpointTest, resume_from_state)
{
    Pipeline p(1);
    p.set_generate_thumbnails(false);

    ASSERT_TRUE(p.saveCheckpoint(test_checkpoint_dir));

    Pipeline p2(1);
    ASSERT_TRUE(p2.loadCheckpoint(test_checkpoint_dir));

    // Should be able to resume from the same or earlier state
    EXPECT_TRUE(p2.resumeFromState(PipelineState::INITIAL_PROCESSING));
    EXPECT_EQ(PipelineState::INITIAL_PROCESSING, p2.getState());
}
