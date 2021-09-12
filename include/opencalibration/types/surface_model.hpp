#pragma once

#include <opencalibration/types/plane.hpp>
#include <opencalibration/types/point_cloud.hpp>

namespace opencalibration
{
struct surface_model
{
    std::vector<point_cloud> cloud;
    std::vector<plane_3_corners_d> mesh;
};
} // namespace opencalibration
