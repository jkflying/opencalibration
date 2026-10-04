#include <opencalibration/io/checkpoint.hpp>
#include <opencalibration/pipeline/pipeline.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

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
    const Eigen::Vector3d precise(1234.567891234, -2345.678912345, 98.7654321);
    cloud1.push_back(precise);
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

    CheckpointData loaded;
    ASSERT_TRUE(loadCheckpoint(test_checkpoint_dir, loaded));

    EXPECT_EQ(data.surfaces.size(), loaded.surfaces.size());
    ASSERT_EQ(2u, loaded.surfaces.size());

    EXPECT_EQ(2u, loaded.surfaces[0].cloud.size());
    ASSERT_EQ(2u, loaded.surfaces[0].cloud[0].size());
    EXPECT_EQ(precise, loaded.surfaces[0].cloud[0][0]);

    EXPECT_EQ(1u, loaded.surfaces[0].cloud[1].size());
    EXPECT_DOUBLE_EQ(7.0, loaded.surfaces[0].cloud[1][0].x());

    EXPECT_EQ(1u, loaded.surfaces[1].cloud.size());
    EXPECT_EQ(1u, loaded.surfaces[1].cloud[0].size());
    EXPECT_DOUBLE_EQ(10.0, loaded.surfaces[1].cloud[0][0].x());
}

TEST_F(CheckpointTest, save_and_load_color_balance)
{
    // GIVEN: a checkpoint holding a solved color balance
    CheckpointData data;
    data.state = PipelineState::GENERATE_GEOTIFF;
    data.color_balance.success = true;
    data.color_balance.final_cost = 1.5;
    data.color_balance.num_iterations = 7;
    data.color_balance.horizontal_view_dir_log_cbrt_gain = {0.01, -0.02};
    data.color_balance.per_image_params[0xFFFFFFFF12345678ull] = {0.1, {1.0, -2.0}, 0.3, {0.04, -0.05}};
    data.color_balance.per_model_params[3] = {{-0.1, 0.02, -0.003}};

    // WHEN: saving and loading it
    ASSERT_TRUE(saveCheckpoint(data, test_checkpoint_dir));
    CheckpointData loaded;
    ASSERT_TRUE(loadCheckpoint(test_checkpoint_dir, loaded));

    // THEN: every color balance parameter survives
    const auto &cb = loaded.color_balance;
    EXPECT_TRUE(cb.success);
    EXPECT_DOUBLE_EQ(1.5, cb.final_cost);
    EXPECT_EQ(7, cb.num_iterations);
    EXPECT_EQ(data.color_balance.horizontal_view_dir_log_cbrt_gain, cb.horizontal_view_dir_log_cbrt_gain);
    ASSERT_EQ(1u, cb.per_image_params.count(0xFFFFFFFF12345678ull));
    const auto &img = cb.per_image_params.at(0xFFFFFFFF12345678ull);
    EXPECT_DOUBLE_EQ(0.1, img.log_cbrt_exposure);
    EXPECT_EQ((std::array<double, 2>{1.0, -2.0}), img.ab_offset);
    EXPECT_DOUBLE_EQ(0.3, img.brdf_coeff);
    EXPECT_EQ((std::array<double, 2>{0.04, -0.05}), img.slope);
    ASSERT_EQ(1u, cb.per_model_params.count(3));
    EXPECT_EQ((std::array<double, 3>{-0.1, 0.02, -0.003}), cb.per_model_params.at(3).log_cbrt_falloff_coeffs);
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

TEST_F(CheckpointTest, load_nonexistent)
{
    CheckpointData data;
    EXPECT_FALSE(loadCheckpoint("/nonexistent/path/to/checkpoint", data));
}

TEST_F(CheckpointTest, fromString_toString_roundtrip)
{
    std::vector<PipelineState> states = {PipelineState::INITIAL_PROCESSING,     PipelineState::INITIAL_GLOBAL_RELAX,
                                         PipelineState::CAMERA_PARAMETER_RELAX, PipelineState::FINAL_GLOBAL_RELAX,
                                         PipelineState::GENERATE_THUMBNAIL,     PipelineState::GENERATE_GEOTIFF,
                                         PipelineState::GENERATE_GEOTIFF,       PipelineState::GENERATE_GEOTIFF,
                                         PipelineState::GENERATE_GEOTIFF,       PipelineState::COMPLETE};

    std::vector<std::string> state_strings = {"INITIAL_PROCESSING",     "INITIAL_GLOBAL_RELAX",
                                              "CAMERA_PARAMETER_RELAX", "FINAL_GLOBAL_RELAX",
                                              "GENERATE_THUMBNAIL",     "GENERATE_GEOTIFF",
                                              "GENERATE_LAYERS",        "BLEND_LAYERS",
                                              "COLOR_BALANCE",          "COMPLETE"};

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

TEST_F(CheckpointTest, resume_from_later_state_fails)
{
    Pipeline p(1);
    p.set_generate_thumbnails(false);

    ASSERT_TRUE(p.saveCheckpoint(test_checkpoint_dir));

    Pipeline p2(1);
    ASSERT_TRUE(p2.loadCheckpoint(test_checkpoint_dir));

    EXPECT_FALSE(p2.resumeFromState(PipelineState::COMPLETE));
}

TEST_F(CheckpointTest, resume_follows_execution_order)
{
    // GIVEN: a checkpoint taken at mesh refinement, which runs directly after initial processing
    CheckpointData data;
    data.state = PipelineState::MESH_REFINEMENT;
    ASSERT_TRUE(saveCheckpoint(data, test_checkpoint_dir));

    // WHEN: we try to resume from a relax stage that has not run yet, or from the stage before
    Pipeline p(1);
    ASSERT_TRUE(p.loadCheckpoint(test_checkpoint_dir));
    const bool resumed_later = p.resumeFromState(PipelineState::CAMERA_PARAMETER_RELAX);
    const bool resumed_earlier = p.resumeFromState(PipelineState::INITIAL_PROCESSING);

    // THEN: only the stage that already ran is accepted
    EXPECT_FALSE(resumed_later);
    EXPECT_TRUE(resumed_earlier);
}

TEST_F(CheckpointTest, load_corrupt_project_db)
{
    std::filesystem::create_directories(test_checkpoint_dir);
    std::ofstream(test_checkpoint_dir + "/project.db") << "not a database";

    CheckpointData data;
    EXPECT_FALSE(loadCheckpoint(test_checkpoint_dir, data));
    EXPECT_TRUE(listCheckpointStages(test_checkpoint_dir).empty());
}

TEST_F(CheckpointTest, stages_store_only_changes)
{
    // GIVEN: a calibrated three image graph saved as a first stage
    Pipeline p(1);
    p.add({TEST_DATA_DIR "IMG_1378_RGB.jpg", TEST_DATA_DIR "IMG_1379_RGB.jpg", TEST_DATA_DIR "IMG_1392_RGB.jpg"});
    while (p.getState() != PipelineState::COMPLETE)
    {
        p.iterateOnce();
    }
    CheckpointData first;
    first.graph = p.getGraph();
    first.state = PipelineState::INITIAL_GLOBAL_RELAX;
    first.surfaces.resize(2);
    first.surfaces[1].cloud.push_back({Eigen::Vector3d(1, 2, 3)});
    ASSERT_GT(first.graph.size_edges(), 0u);
    ASSERT_TRUE(saveCheckpoint(first, test_checkpoint_dir));

    // WHEN: a second stage moves a node, drops an edge and a surface, and a third repeats it unchanged
    CheckpointData second;
    second.graph = first.graph;
    second.state = PipelineState::COMPLETE;
    second.graph.nodebegin()->second.payload.position.x() += 1;
    second.graph.removeEdge(second.graph.cedgebegin()->first);
    second.surfaces.resize(1);
    ASSERT_TRUE(saveCheckpoint(second, test_checkpoint_dir));
    const auto db_size = std::filesystem::file_size(test_checkpoint_dir + "/project.db");
    ASSERT_TRUE(saveCheckpoint(second, test_checkpoint_dir));
    EXPECT_LT(std::filesystem::file_size(test_checkpoint_dir + "/project.db"), db_size + 16384);

    // THEN: every stage loads back as it was saved, features included
    const auto stages = listCheckpointStages(test_checkpoint_dir);
    ASSERT_EQ(3u, stages.size());
    for (const auto &[stage, expected] :
         {std::make_pair(stages[0].name, &first), {stages[1].name, &second}, {stages[2].name, &second}})
    {
        CheckpointData loaded;
        ASSERT_TRUE(loadCheckpoint(test_checkpoint_dir, loaded, stage));
        EXPECT_TRUE(loaded.graph == expected->graph) << stage;
        EXPECT_EQ(expected->state, loaded.state);
        ASSERT_EQ(expected->surfaces.size(), loaded.surfaces.size());
        EXPECT_EQ(expected->surfaces.back().cloud, loaded.surfaces.back().cloud);
    }
}
