#pragma once

#include <stddef.h>

namespace opencalibration
{
struct CameraModel
{
    size_t pixels_rows = 0;
    size_t pixels_cols = 0;

    double focal_length_pixels = 0;

    enum class ProjectionType
    {
        PLANAR
    } projection_type = ProjectionType::PLANAR;
};
} // namespace opencalibration
