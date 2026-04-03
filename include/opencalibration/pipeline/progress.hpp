#pragma once

#include <opencalibration/types/pipeline_state.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace opencalibration
{

struct TileThumbnail
{
    std::string png_base64;
    double bounds_min_x = 0;
    double bounds_max_y = 0;
    double meters_per_pixel = 0;
};

struct TileUpdate
{
    int pixel_x = 0;
    int pixel_y = 0;
    int pixel_w = 0;
    int pixel_h = 0;
    int total_output_width = 0;
    int total_output_height = 0;
    int tile_index = 0;
    int total_tiles = 1;
    TileThumbnail thumbnail;
};

struct StepCompletionInfo
{
    std::vector<size_t> loaded_ids, linked_ids;
    std::vector<std::vector<size_t>> relaxed_ids;
    size_t images_loaded = 0;
    size_t queue_size_remaining = 0;
    PipelineState state;
    uint64_t state_iteration = 0;
    std::string activity;
    float global_fraction = 0.f;
    float local_fraction = 0.f;
    bool surfaces_updated = false;
    std::optional<TileUpdate> tile_update;
};

using StepCompletionCallback = std::function<void(const StepCompletionInfo &)>;
using TileProgressCallback = std::function<void(const TileUpdate &)>;

} // namespace opencalibration
