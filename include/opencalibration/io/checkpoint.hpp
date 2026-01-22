#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/pipeline_state.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <string>

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
};

bool saveCheckpoint(const CheckpointData &data, const std::string &checkpoint_dir);
bool loadCheckpoint(const std::string &checkpoint_dir, CheckpointData &data);
bool validateCheckpoint(const std::string &checkpoint_dir);

} // namespace opencalibration
