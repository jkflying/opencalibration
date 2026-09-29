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

TEST_F(CheckpointTest, load_ignores_wrongly_typed_metadata)
{
    // GIVEN: a saved checkpoint whose metadata was edited to hold fields of the wrong JSON type
    CheckpointData data;
    data.state = PipelineState::FINAL_GLOBAL_RELAX;
    ASSERT_TRUE(saveCheckpoint(data, test_checkpoint_dir));
    {
        std::ofstream out(std::filesystem::path(test_checkpoint_dir) / "00_FINAL_GLOBAL_RELAX_metadata.json");
        out << R"({"version":2,"state":5,"state_run_count":"x","origin_latitude":"north","origin_longitude":[],)"
               R"("surface_count":-1,"color_balance":{"success":1,"final_cost":"x","num_iterations":1.5,)"
               R"("horizontal_view_dir_log_cbrt_gain":"x","images":[{"id":"a"},7],"models":{}}})";
    }

    // WHEN: we load it
    CheckpointData loaded;
    const bool ok = loadCheckpoint(test_checkpoint_dir, loaded);

    // THEN: loading succeeds and the malformed fields keep their defaults
    EXPECT_TRUE(ok);
    EXPECT_EQ(loaded.origin_latitude, 0.0);
    EXPECT_TRUE(loaded.color_balance.per_image_params.empty());
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

TEST_F(CheckpointTest, load_malformed_metadata)
{
    std::filesystem::create_directories(test_checkpoint_dir);
    {
        std::ofstream out(test_checkpoint_dir + "/metadata.json");
        out << "not valid json{{{";
    }
    {
        std::ofstream out(test_checkpoint_dir + "/graph.json");
        out << "{}";
    }

    CheckpointData data;
    EXPECT_FALSE(loadCheckpoint(test_checkpoint_dir, data));
}

TEST_F(CheckpointTest, load_wrong_version)
{
    std::filesystem::create_directories(test_checkpoint_dir);
    {
        std::ofstream out(test_checkpoint_dir + "/metadata.json");
        out << R"({"version": 999})";
    }
    {
        std::ofstream out(test_checkpoint_dir + "/graph.json");
        out << "{}";
    }

    CheckpointData data;
    EXPECT_FALSE(loadCheckpoint(test_checkpoint_dir, data));
}

TEST_F(CheckpointTest, load_missing_graph)
{
    std::filesystem::create_directories(test_checkpoint_dir);
    {
        std::ofstream out(test_checkpoint_dir + "/metadata.json");
        out << R"({"version": 1, "state": "INITIAL_PROCESSING", "state_run_count": 0, "origin_latitude": 0, "origin_longitude": 0, "surface_count": 0})";
    }

    CheckpointData data;
    EXPECT_FALSE(loadCheckpoint(test_checkpoint_dir, data));
}

TEST_F(CheckpointTest, stages_share_features_file)
{
    Pipeline p(1);
    p.add({TEST_DATA_DIR "P2530253.JPG"});
    while (p.getState() != PipelineState::COMPLETE)
    {
        p.iterateOnce();
    }
    ASSERT_GT(p.getGraph().cnodebegin()->second.payload.features.size(), 0u);

    ASSERT_TRUE(p.saveCheckpoint(test_checkpoint_dir));
    const auto features_path = std::filesystem::path(test_checkpoint_dir) / "features.json.zst";
    const auto features_written = std::filesystem::last_write_time(features_path);
    ASSERT_TRUE(p.saveCheckpoint(test_checkpoint_dir));
    EXPECT_EQ(features_written, std::filesystem::last_write_time(features_path));

    const auto stages = listCheckpointStages(test_checkpoint_dir);
    ASSERT_EQ(2u, stages.size());
    EXPECT_NE(stages[0].name, stages[1].name);
    EXPECT_EQ(PipelineState::COMPLETE, stages[0].state);

    CheckpointData loaded;
    ASSERT_TRUE(loadCheckpoint(test_checkpoint_dir, loaded, stages[0].name));
    EXPECT_TRUE(loaded.graph == p.getGraph());
}
