#pragma once

#include <opencalibration/types/raster.hpp>

#include <functional>

namespace opencalibration
{

struct TileUpdate
{
    int pixel_x = 0;
    int pixel_y = 0;
    int pixel_w = 0;
    int pixel_h = 0;
    int total_output_width = 0;
    int total_output_height = 0;
    int tile_index = 0;  // 1-based count of tiles completed so far
    int total_tiles = 1; // total number of tiles in this pass
    RGBRaster thumbnail; // max 128x128 downsampled preview (BGR channel order)
};

using TileProgressCallback = std::function<void(const TileUpdate &)>;

} // namespace opencalibration
