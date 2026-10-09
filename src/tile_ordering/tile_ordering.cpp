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

std::vector<ImageLoad> planImageLoads(const std::vector<std::vector<size_t>> &tile_images, size_t capacity,
                                      size_t lookahead)
{
    ankerl::unordered_dense::map<size_t, std::vector<size_t>> uses;
    for (size_t tile = 0; tile < tile_images.size(); tile++)
        for (size_t image : tile_images[tile])
            uses[image].push_back(tile);
    auto next_use = [&](size_t image, size_t from) {
        const auto &image_uses = uses.at(image);
        auto it = std::lower_bound(image_uses.begin(), image_uses.end(), from);
        return it == image_uses.end() ? NO_IMAGE : *it;
    };

    ankerl::unordered_dense::map<size_t, size_t> last_use;
    auto pick_victim = [&](size_t tile) {
        for (size_t window_start = tile - std::min(tile, lookahead);; window_start++)
        {
            auto victim = std::max_element(last_use.begin(), last_use.end(), [&](const auto &a, const auto &b) {
                return next_use(a.first, window_start) < next_use(b.first, window_start);
            });
            if (next_use(victim->first, window_start) > tile)
                return victim;
        }
    };

    std::vector<ImageLoad> loads;
    for (size_t tile = 0; tile < tile_images.size(); tile++)
    {
        const size_t resident_limit = std::max(capacity, tile_images[tile].size());
        for (size_t image : tile_images[tile])
        {
            if (!last_use.contains(image))
            {
                ImageLoad load{tile, image};
                if (last_use.size() >= resident_limit)
                {
                    const auto victim = pick_victim(tile);
                    load.evict = victim->first;
                    load.tiles_done_before_start = victim->second + 1;
                    last_use.erase(victim);
                }
                loads.push_back(load);
            }
            last_use[image] = tile;
        }
    }
    return loads;
}

ImageCacheSettings computeImageCacheSettings(const TileCameraMap &tile_cameras)
{
    constexpr size_t kMedianMultiple = 4;
    constexpr size_t kLookaheadDivisor = 2;
    constexpr size_t kMinSize = 10;
    constexpr size_t kMaxSize = 96;
    constexpr size_t kMinLookahead = 1;

    std::vector<size_t> counts;
    counts.reserve(tile_cameras.size());
    for (const auto &[tile, cams] : tile_cameras)
        if (!cams.empty())
            counts.push_back(cams.size());
    if (counts.empty())
        return {kMinSize, kMinLookahead};

    auto mid = counts.begin() + static_cast<std::ptrdiff_t>(counts.size() / 2);
    std::nth_element(counts.begin(), mid, counts.end());
    return {std::clamp(kMedianMultiple * *mid, kMinSize, kMaxSize), std::max(kMinLookahead, *mid / kLookaheadDivisor)};
}

} // namespace opencalibration
