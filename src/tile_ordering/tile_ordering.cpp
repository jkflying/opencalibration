#include <opencalibration/tile_ordering/tile_ordering.hpp>

#include <opencalibration/types/hilbert.hpp>

#include <algorithm>

namespace opencalibration
{

std::vector<std::pair<int, int>> hilbertTileOrder(int num_tiles_x, int num_tiles_y)
{
    int max_dim = std::max(num_tiles_x, num_tiles_y);
    int order = 1;
    while (order < max_dim)
        order *= 2;

    std::vector<std::pair<uint32_t, std::pair<int, int>>> tiles;
    tiles.reserve(num_tiles_x * num_tiles_y);
    for (int ty = 0; ty < num_tiles_y; ty++)
    {
        for (int tx = 0; tx < num_tiles_x; tx++)
        {
            tiles.push_back({xy2d(order, tx, ty), {tx, ty}});
        }
    }
    std::sort(tiles.begin(), tiles.end());

    std::vector<std::pair<int, int>> result;
    result.reserve(tiles.size());
    for (auto &t : tiles)
        result.push_back(t.second);
    return result;
}

ImageUseSchedule::ImageUseSchedule(const std::vector<std::pair<int, int>> &tile_order,
                                   const TileCameraMap &tile_cameras, int num_tiles_x)
    : num_tiles_x_(num_tiles_x)
{
    for (size_t i = 0; i < tile_order.size(); i++)
    {
        auto it = tile_cameras.find(tileIndex(tile_order[i]));
        if (it == tile_cameras.end())
            continue;
        for (size_t cam : it->second)
            uses_[cam].push_back(i);
    }
}

size_t ImageUseSchedule::nextUse(size_t cam, size_t position) const
{
    auto it = uses_.find(cam);
    if (it == uses_.end())
        return SIZE_MAX;
    auto next = std::lower_bound(it->second.begin(), it->second.end(), position);
    return next == it->second.end() ? SIZE_MAX : *next;
}

size_t ImageUseSchedule::tileIndex(const std::pair<int, int> &tile) const
{
    return static_cast<size_t>(tile.second) * num_tiles_x_ + tile.first;
}

size_t computeImageCacheSize(const TileCameraMap &tile_cameras)
{
    constexpr size_t kMedianMultiple = 7;
    constexpr size_t kMinSize = 10;
    constexpr size_t kMaxSize = 96;

    std::vector<size_t> counts;
    counts.reserve(tile_cameras.size());
    for (const auto &[tile, cams] : tile_cameras)
        if (!cams.empty())
            counts.push_back(cams.size());
    if (counts.empty())
        return kMinSize;

    auto mid = counts.begin() + static_cast<std::ptrdiff_t>(counts.size() / 2);
    std::nth_element(counts.begin(), mid, counts.end());
    size_t max_count = *std::max_element(counts.begin(), counts.end());

    size_t size = std::max({kMedianMultiple * *mid, 2 * max_count, kMinSize});
    return std::min(size, kMaxSize);
}

} // namespace opencalibration
