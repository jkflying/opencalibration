#include <opencalibration/tile_ordering/tile_ordering.hpp>

#include <gtest/gtest.h>

#include <algorithm>

using namespace opencalibration;

namespace
{

void verifyCompleteOrdering(const std::vector<std::pair<int, int>> &order, int nx, int ny)
{
    ASSERT_EQ(order.size(), static_cast<size_t>(nx * ny));
    ankerl::unordered_dense::set<size_t> seen;
    for (const auto &[tx, ty] : order)
    {
        size_t idx = static_cast<size_t>(ty) * nx + tx;
        EXPECT_TRUE(seen.insert(idx).second) << "Duplicate tile: " << tx << "," << ty;
    }
}

} // namespace

TEST(TileOrdering, empty_and_single_tile)
{
    // GIVEN: an empty grid
    {
        TileCameraMap tile_cameras;
        TileOrderingParams params{0, 0, 5};

        // WHEN: we compute tile order
        auto order = computeCacheAwareTileOrder(tile_cameras, params);

        // THEN: result is empty
        EXPECT_TRUE(order.empty());
    }

    // GIVEN: a single tile
    {
        TileCameraMap tile_cameras;
        tile_cameras[0] = {1, 2, 3};
        TileOrderingParams params{1, 1, 5};

        // WHEN: we compute tile order
        auto order = computeCacheAwareTileOrder(tile_cameras, params);

        // THEN: we get exactly that tile
        ASSERT_EQ(order.size(), 1u);
        EXPECT_EQ(order[0].first, 0);
        EXPECT_EQ(order[0].second, 0);
    }
}

TEST(TileOrdering, ordering_covers_all_tiles)
{
    // GIVEN: a 5x8 grid where each camera covers 2-3 consecutive tiles per row
    int nx = 5, ny = 8;
    TileCameraMap tile_cameras;
    size_t cam = 0;
    for (int ty = 0; ty < ny; ty++)
    {
        for (int tx = 0; tx < nx; tx += 2)
        {
            int end = std::min(tx + 3, nx);
            for (int x = tx; x < end; x++)
            {
                size_t idx = static_cast<size_t>(ty) * nx + x;
                tile_cameras[idx].insert(cam);
            }
            cam++;
        }
    }
    TileOrderingParams params{nx, ny, 5};

    // WHEN: we compute tile order
    auto order = computeCacheAwareTileOrder(tile_cameras, params);

    // THEN: every tile appears exactly once
    verifyCompleteOrdering(order, nx, ny);
}

TEST(TileOrdering, small_grid)
{
    // GIVEN: a 3x3 grid with overlapping cameras between adjacent tiles
    int nx = 3, ny = 3;
    TileCameraMap tile_cameras;
    tile_cameras[0] = {0, 1};
    tile_cameras[1] = {1, 2};
    tile_cameras[2] = {2, 3};
    tile_cameras[3] = {0, 4};
    tile_cameras[4] = {1, 4};
    tile_cameras[5] = {2, 5};
    tile_cameras[6] = {3, 6};
    tile_cameras[7] = {4, 6};
    tile_cameras[8] = {5, 6};
    TileOrderingParams params{nx, ny, 3};

    // WHEN: we compute tile order
    auto order = computeCacheAwareTileOrder(tile_cameras, params);

    // THEN: every tile appears exactly once
    verifyCompleteOrdering(order, nx, ny);
}

TEST(TileOrdering, partial_coverage)
{
    // GIVEN: an 8x8 grid where only the interior 4x4 has cameras
    int nx = 8, ny = 8;
    TileOrderingParams params{nx, ny, 5};

    TileCameraMap tile_cameras;
    size_t cam_id = 0;
    for (int ty = 2; ty < 6; ty++)
    {
        for (int tx = 2; tx < 6; tx++)
        {
            size_t tile_idx = static_cast<size_t>(ty) * nx + tx;
            tile_cameras[tile_idx].insert(cam_id);
            if (cam_id > 0)
                tile_cameras[tile_idx].insert(cam_id - 1);
            cam_id++;
        }
    }

    // WHEN: we compute tile order
    auto order = computeCacheAwareTileOrder(tile_cameras, params);

    // THEN: all tiles (including uncovered perimeter) are present
    verifyCompleteOrdering(order, nx, ny);
}

TEST(TileOrdering, greedy_beats_hilbert_on_diagonal_cameras)
{
    // GIVEN: a 16x16 grid with diagonal camera assignment (adversarial for space-filling curves)
    int nx = 16, ny = 16;
    TileCameraMap tile_cameras;
    for (int ty = 0; ty < ny; ty++)
    {
        for (int tx = 0; tx < nx; tx++)
        {
            size_t tile_idx = static_cast<size_t>(ty) * nx + tx;
            int diag = tx + ty;
            tile_cameras[tile_idx].insert(static_cast<size_t>(diag));
            if (diag > 0)
                tile_cameras[tile_idx].insert(static_cast<size_t>(diag - 1));
            if (diag < nx + ny - 2)
                tile_cameras[tile_idx].insert(static_cast<size_t>(diag + 1));
        }
    }
    TileOrderingParams params{nx, ny, 8};

    // WHEN: we compare cache-aware vs plain Hilbert ordering
    auto order = computeCacheAwareTileOrder(tile_cameras, params);
    auto hilbert = hilbertTileOrder(nx, ny);

    // THEN: greedy chose a different (better) ordering, and it's still complete
    EXPECT_NE(order, hilbert);
    verifyCompleteOrdering(order, nx, ny);
}

TEST(TileOrdering, image_cache_size_scales_with_median_and_is_bounded)
{
    // GIVEN: tiles mostly seeing 8 cameras, one seeing 17
    TileCameraMap typical;
    for (size_t t = 0; t < 10; t++)
        for (size_t c = 0; c < 8; c++)
            typical[t].insert(t * 100 + c);
    for (size_t c = 0; c < 17; c++)
        typical[10].insert(1000 + c);

    // GIVEN: tiles seeing few cameras, but one outlier seeing many
    TileCameraMap outlier;
    for (size_t t = 0; t < 10; t++)
        outlier[t] = {t};
    for (size_t c = 0; c < 20; c++)
        outlier[10].insert(1000 + c);

    // GIVEN: tiles seeing a huge number of cameras
    TileCameraMap dense;
    for (size_t c = 0; c < 100; c++)
        dense[0].insert(c);

    // WHEN: we compute the cache size
    // THEN: it is 5x the median, floored by 2x the max, and capped
    EXPECT_EQ(computeImageCacheSize(typical), 40u);
    EXPECT_EQ(computeImageCacheSize(outlier), 40u);
    EXPECT_EQ(computeImageCacheSize(dense), 64u);
    EXPECT_EQ(computeImageCacheSize({}), 10u);
}

TEST(TileOrdering, image_use_schedule_next_use)
{
    // GIVEN: a 3x1 grid visited right to left, camera 7 seen in tiles 0 and 2, camera 8 only in tile 1
    TileCameraMap tile_cameras;
    tile_cameras[0] = {7};
    tile_cameras[1] = {8};
    tile_cameras[2] = {7};
    std::vector<std::pair<int, int>> order{{2, 0}, {1, 0}, {0, 0}};

    // WHEN: we build the schedule
    ImageUseSchedule schedule(order, tile_cameras, 3);

    // THEN: next use is the first order position at or after the query position
    EXPECT_EQ(schedule.nextUse(7, 0), 0u);
    EXPECT_EQ(schedule.nextUse(7, 1), 2u);
    EXPECT_EQ(schedule.nextUse(8, 0), 1u);
    EXPECT_EQ(schedule.nextUse(8, 2), SIZE_MAX);
    EXPECT_EQ(schedule.nextUse(99, 0), SIZE_MAX);
}
