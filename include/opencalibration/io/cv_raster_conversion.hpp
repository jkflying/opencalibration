#pragma once

#include <opencalibration/types/raster.hpp>
#include <opencv2/core/mat.hpp>

namespace opencalibration
{
cv::Mat rasterToCv(const GenericRaster &raster);
cv::Mat rasterToCv(const GenericLayer &layer);
GenericRaster cvToRaster(const cv::Mat &mat);
RGBRaster RasterToRGB(const GenericRaster &raster);
} // namespace opencalibration
