#pragma once

#include <ankerl/unordered_dense.h>
#include <cstddef>
#include <utility>
#include <vector>

namespace opencalibration
{

using TileCameraMap = ankerl::unordered_dense::map<size_t, ankerl::unordered_dense::set<size_t>>;

struct TileOrderingParams
{
    int num_tiles_x;
    int num_tiles_y;
    size_t cache_size;
};

std::vector<std::pair<int, int>> computeCacheAwareTileOrder(const TileCameraMap &tile_cameras,
                                                            const TileOrderingParams &params);

std::vector<std::pair<int, int>> hilbertTileOrder(int num_tiles_x, int num_tiles_y);

} // namespace opencalibration
