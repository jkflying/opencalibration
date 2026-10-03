#pragma once

#include <opencalibration/ortho/color_balance.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/pipeline_state.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <string>

struct sqlite3;

namespace opencalibration
{

struct CheckpointData
{
    MeasurementGraph graph;
    std::vector<surface_model> surfaces;
    double origin_latitude = 0.0;
    double origin_longitude = 0.0;
    PipelineState state = PipelineState::INITIAL_PROCESSING;
    uint64_t state_run_count = 0;
    orthomosaic::ColorBalanceResult color_balance;
};

struct CheckpointStage
{
    std::string name;
    PipelineState state;
};

class ProjectStore : public std::enable_shared_from_this<ProjectStore>
{
  public:
    static std::shared_ptr<ProjectStore> open(const std::string &dir);
    ~ProjectStore();

    std::optional<image> loadImage(const std::string &path);
    bool saveImage(image &img);

    std::vector<CheckpointStage> stages();
    bool save(const CheckpointData &data);
    bool load(CheckpointData &data, const std::string &stage);

  private:
    explicit ProjectStore(sqlite3 *db);
    bool exec(const char *sql);
    bool putImage(const std::string &path, const std::string &stamp, const image &img,
                  const std::vector<feature_2d> &features);
    std::vector<feature_2d> readFeatures(const std::string &path);
    FeatureSet storedFeatures(const std::string &path, size_t size);

    sqlite3 *_db;
    std::recursive_mutex _mutex;
};

std::vector<CheckpointStage> listCheckpointStages(const std::string &checkpoint_dir);
bool saveCheckpoint(const CheckpointData &data, const std::string &checkpoint_dir);
bool loadCheckpoint(const std::string &checkpoint_dir, CheckpointData &data, std::string stage = "");

} // namespace opencalibration
