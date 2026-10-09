#pragma once

#include <ankerl/unordered_dense.h>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace opencalibration
{

using TileCameraMap = ankerl::unordered_dense::map<size_t, ankerl::unordered_dense::set<size_t>>;

std::vector<std::pair<int, int>> hilbertTileOrder(int num_tiles_x, int num_tiles_y);

constexpr size_t NO_IMAGE = SIZE_MAX;

struct ImageLoad
{
    size_t tile;
    size_t image;
    size_t evict = NO_IMAGE;
    size_t tiles_done_before_start = 0;
};

std::vector<ImageLoad> planImageLoads(const std::vector<std::vector<size_t>> &tile_images, size_t capacity,
                                      size_t lookahead);

struct ImageCacheSettings
{
    size_t capacity;
    size_t lookahead;
};

ImageCacheSettings computeImageCacheSettings(const TileCameraMap &tile_cameras);

} // namespace opencalibration
