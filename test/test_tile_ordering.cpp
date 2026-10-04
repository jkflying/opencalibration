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
    // GIVEN: an empty grid and a single-tile grid
    // WHEN: we compute tile orders
    auto empty = hilbertTileOrder(0, 0);
    auto single = hilbertTileOrder(1, 1);

    // THEN: the empty grid has no tiles and the single grid has exactly its one tile
    EXPECT_TRUE(empty.empty());
    ASSERT_EQ(single.size(), 1u);
    EXPECT_EQ(single[0], std::make_pair(0, 0));
}

TEST(TileOrdering, ordering_covers_all_tiles)
{
    // GIVEN: a non-square, non-power-of-two grid
    int nx = 5, ny = 8;

    // WHEN: we compute tile order
    auto order = hilbertTileOrder(nx, ny);

    // THEN: every tile appears exactly once
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
    // THEN: it is 7x the median, floored by 2x the max, and capped
    EXPECT_EQ(computeImageCacheSize(typical), 56u);
    EXPECT_EQ(computeImageCacheSize(outlier), 40u);
    EXPECT_EQ(computeImageCacheSize(dense), 96u);
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
