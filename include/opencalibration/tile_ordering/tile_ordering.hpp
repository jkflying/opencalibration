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
    size_t free_loads_per_tile = 0;
};

std::vector<std::pair<int, int>> computeCacheAwareTileOrder(const TileCameraMap &tile_cameras,
                                                            const TileOrderingParams &params);

std::vector<std::pair<int, int>> hilbertTileOrder(int num_tiles_x, int num_tiles_y);

class ImageUseSchedule
{
  public:
    ImageUseSchedule(const std::vector<std::pair<int, int>> &tile_order, const TileCameraMap &tile_cameras,
                     int num_tiles_x);

    [[nodiscard]] size_t nextUse(size_t cam, size_t position) const;
    [[nodiscard]] size_t tileIndex(const std::pair<int, int> &tile) const;

  private:
    ankerl::unordered_dense::map<size_t, std::vector<size_t>> uses_;
    int num_tiles_x_;
};

size_t computeImageCacheSize(const TileCameraMap &tile_cameras);

} // namespace opencalibration
