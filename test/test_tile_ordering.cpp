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

TEST(TileOrdering, image_cache_settings_scale_with_median_and_are_bounded)
{
    // GIVEN: tiles mostly seeing 8 cameras, one seeing 17
    TileCameraMap typical;
    for (size_t t = 0; t < 10; t++)
        for (size_t c = 0; c < 8; c++)
            typical[t].insert(t * 100 + c);
    for (size_t c = 0; c < 17; c++)
        typical[10].insert(1000 + c);

    // GIVEN: tiles seeing a huge number of cameras
    TileCameraMap dense;
    for (size_t c = 0; c < 100; c++)
        dense[0].insert(c);

    // WHEN: we compute the cache settings
    const auto typical_settings = computeImageCacheSettings(typical);
    const auto dense_settings = computeImageCacheSettings(dense);
    const auto empty_settings = computeImageCacheSettings({});

    // THEN: capacity is 4x the median within bounds, and lookahead is half the median
    EXPECT_EQ(typical_settings.capacity, 32u);
    EXPECT_EQ(typical_settings.lookahead, 4u);
    EXPECT_EQ(dense_settings.capacity, 96u);
    EXPECT_EQ(empty_settings.capacity, 10u);
    EXPECT_EQ(empty_settings.lookahead, 1u);
}

TEST(TileOrdering, plan_evicts_image_needed_furthest_away_at_time_of_use)
{
    // GIVEN: room for 2 images, where image 1 is needed at tiles 0-1 and image 3 at tiles 0 and 3
    const std::vector<std::vector<size_t>> tile_images{{1, 3}, {1}, {2}, {3}};

    // WHEN: planning the loads
    const auto plan = planImageLoads(tile_images, 2, 0);

    // THEN: image 2 replaces image 1 once tile 1 is done, so image 3 is never reloaded
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan[2].tile, 2u);
    EXPECT_EQ(plan[2].image, 2u);
    EXPECT_EQ(plan[2].evict, 1u);
    EXPECT_EQ(plan[2].tiles_done_before_start, 2u);
    EXPECT_EQ(plan[0].evict, NO_IMAGE);
    EXPECT_EQ(plan[1].evict, NO_IMAGE);
}

TEST(TileOrdering, plan_reloads_when_capacity_is_exceeded_and_grows_for_large_tiles)
{
    // GIVEN: room for 1 image, alternating between two images, then a tile needing three
    const std::vector<std::vector<size_t>> tile_images{{1}, {2}, {1}, {4, 5, 6}};

    // WHEN: planning the loads
    const auto plan = planImageLoads(tile_images, 1, 0);

    // THEN: every tile change reloads, and the large tile grows the cache instead of evicting its own images
    ASSERT_EQ(plan.size(), 6u);
    EXPECT_EQ(plan[1].evict, 1u);
    EXPECT_EQ(plan[2].evict, 2u);
    EXPECT_EQ(plan[3].evict, NO_IMAGE);
    EXPECT_EQ(plan[4].evict, NO_IMAGE);
    EXPECT_EQ(plan[5].evict, 1u);
}

TEST(TileOrdering, plan_lookahead_evicts_an_image_unused_in_the_window_so_loads_start_early)
{
    // GIVEN: the same tiles as the Belady case, but with one tile of lookahead
    const std::vector<std::vector<size_t>> tile_images{{1, 3}, {1}, {2}, {3}};

    // WHEN: planning the loads
    const auto plan = planImageLoads(tile_images, 2, 1);

    // THEN: image 3 is evicted instead of image 1 (still needed at tile 1), so image 2 can load during tile 1,
    // at the cost of reloading image 3
    ASSERT_EQ(plan.size(), 4u);
    EXPECT_EQ(plan[2].image, 2u);
    EXPECT_EQ(plan[2].evict, 3u);
    EXPECT_EQ(plan[2].tiles_done_before_start, 1u);
    EXPECT_EQ(plan[3].image, 3u);
}
