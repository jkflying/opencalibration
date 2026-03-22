#include <opencalibration/tile_ordering/tile_ordering.hpp>

#include <opencalibration/types/hilbert.hpp>

#include <spdlog/spdlog.h>

#include <algorithm>

namespace opencalibration
{

namespace
{

struct LRUCache
{
    std::vector<size_t> entries;
    size_t capacity;

    explicit LRUCache(size_t cap) : capacity(cap)
    {
        entries.reserve(cap);
    }

    void touch(size_t cam)
    {
        auto it = std::find(entries.begin(), entries.end(), cam);
        if (it != entries.end())
        {
            entries.erase(it);
            entries.insert(entries.begin(), cam);
        }
        else
        {
            if (entries.size() >= capacity)
                entries.pop_back();
            entries.insert(entries.begin(), cam);
        }
    }

    bool contains(size_t cam) const
    {
        return std::find(entries.begin(), entries.end(), cam) != entries.end();
    }
};

std::pair<std::vector<std::pair<int, int>>, size_t> cacheAwareSearch(const TileCameraMap &tile_cameras,
                                                                     const TileOrderingParams &params)
{
    int total = params.num_tiles_x * params.num_tiles_y;

    std::vector<size_t> covered_tiles;
    std::vector<size_t> uncovered_tiles;
    for (int ty = 0; ty < params.num_tiles_y; ty++)
    {
        for (int tx = 0; tx < params.num_tiles_x; tx++)
        {
            size_t idx = static_cast<size_t>(ty) * params.num_tiles_x + tx;
            auto it = tile_cameras.find(idx);
            if (it != tile_cameras.end() && !it->second.empty())
                covered_tiles.push_back(idx);
            else
                uncovered_tiles.push_back(idx);
        }
    }

    if (covered_tiles.empty())
    {
        std::vector<std::pair<int, int>> result;
        result.reserve(uncovered_tiles.size());
        for (size_t idx : uncovered_tiles)
            result.push_back({static_cast<int>(idx % params.num_tiles_x),
                              static_cast<int>(idx / params.num_tiles_x)});
        return {std::move(result), 0};
    }

    ankerl::unordered_dense::map<size_t, std::vector<size_t>> camera_to_tiles;
    for (size_t tile_idx : covered_tiles)
    {
        for (size_t cam : tile_cameras.at(tile_idx))
            camera_to_tiles[cam].push_back(tile_idx);
    }

    auto greedySearch = [&](size_t start_tile) -> std::pair<std::vector<size_t>, size_t> {
        LRUCache cache(params.cache_size);
        std::vector<bool> visited(total, false);
        std::vector<size_t> order;
        order.reserve(covered_tiles.size());
        size_t total_misses = 0;
        const ankerl::unordered_dense::set<size_t> *last_cams = nullptr;

        auto visit = [&](size_t tile_idx) {
            visited[tile_idx] = true;
            order.push_back(tile_idx);
            auto it = tile_cameras.find(tile_idx);
            if (it != tile_cameras.end())
            {
                last_cams = &it->second;
                for (size_t cam : it->second)
                {
                    if (!cache.contains(cam))
                        total_misses++;
                    cache.touch(cam);
                }
            }
            else
            {
                last_cams = nullptr;
            }
        };

        visit(start_tile);

        while (order.size() < covered_tiles.size())
        {
            ankerl::unordered_dense::set<size_t> neighborhood;
            for (size_t cam : cache.entries)
            {
                auto cam_it = camera_to_tiles.find(cam);
                if (cam_it == camera_to_tiles.end())
                    continue;
                for (size_t tile_idx : cam_it->second)
                {
                    if (!visited[tile_idx])
                        neighborhood.insert(tile_idx);
                }
            }

            size_t best_tile = SIZE_MAX;

            if (!neighborhood.empty())
            {
                size_t best_misses = SIZE_MAX;
                size_t best_continuity = 0;
                for (size_t tile_idx : neighborhood)
                {
                    const auto &cams = tile_cameras.at(tile_idx);
                    size_t misses = 0;
                    size_t continuity = 0;
                    for (size_t cam : cams)
                    {
                        if (!cache.contains(cam))
                            misses++;
                        if (last_cams && last_cams->count(cam))
                            continuity++;
                    }
                    if (misses < best_misses || (misses == best_misses && continuity > best_continuity))
                    {
                        best_misses = misses;
                        best_continuity = continuity;
                        best_tile = tile_idx;
                    }
                }
            }

            if (best_tile == SIZE_MAX)
            {
                size_t best_count = 0;
                for (size_t tile_idx : covered_tiles)
                {
                    if (visited[tile_idx])
                        continue;
                    size_t count = tile_cameras.at(tile_idx).size();
                    if (count > best_count)
                    {
                        best_count = count;
                        best_tile = tile_idx;
                    }
                }
            }

            visit(best_tile);
        }

        return {std::move(order), total_misses};
    };

    size_t start_tile = covered_tiles[0];
    size_t max_cams = 0;
    for (size_t tile_idx : covered_tiles)
    {
        size_t count = tile_cameras.at(tile_idx).size();
        if (count > max_cams)
        {
            max_cams = count;
            start_tile = tile_idx;
        }
    }

    auto [best_order, best_misses] = greedySearch(start_tile);

    std::vector<std::pair<int, int>> result;
    result.reserve(total);
    for (size_t tile_idx : best_order)
        result.push_back({static_cast<int>(tile_idx % params.num_tiles_x),
                          static_cast<int>(tile_idx / params.num_tiles_x)});
    for (size_t tile_idx : uncovered_tiles)
        result.push_back({static_cast<int>(tile_idx % params.num_tiles_x),
                          static_cast<int>(tile_idx / params.num_tiles_x)});

    return {std::move(result), best_misses};
}

size_t simulateCacheMisses(const std::vector<std::pair<int, int>> &tile_order, const TileCameraMap &tile_cameras,
                           const TileOrderingParams &params)
{
    LRUCache cache(params.cache_size);
    size_t misses = 0;

    for (const auto &[tx, ty] : tile_order)
    {
        size_t tile_idx = static_cast<size_t>(ty) * params.num_tiles_x + tx;
        auto it = tile_cameras.find(tile_idx);
        if (it == tile_cameras.end())
            continue;

        for (size_t cam : it->second)
        {
            if (!cache.contains(cam))
                misses++;
            cache.touch(cam);
        }
    }
    return misses;
}

} // namespace

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

std::vector<std::pair<int, int>> computeCacheAwareTileOrder(const TileCameraMap &tile_cameras,
                                                            const TileOrderingParams &params)
{
    int total = params.num_tiles_x * params.num_tiles_y;
    if (total == 0)
        return {};

    auto [greedy, greedy_misses] = cacheAwareSearch(tile_cameras, params);
    auto hilbert = hilbertTileOrder(params.num_tiles_x, params.num_tiles_y);

    size_t hilbert_misses = simulateCacheMisses(hilbert, tile_cameras, params);

    spdlog::debug("tile ordering: greedy={}, hilbert={}", greedy_misses, hilbert_misses);
    return greedy_misses <= hilbert_misses ? std::move(greedy) : std::move(hilbert);
}

} // namespace opencalibration
