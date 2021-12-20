#pragma once

#include <opencalibration/types/mesh_graph.hpp>
#include <opencalibration/types/plane.hpp>
#include <opencalibration/types/point_cloud.hpp>

namespace opencalibration
{
struct surface_model
{
    std::vector<point_cloud> cloud;
    MeshGraph mesh;
};
} // namespace opencalibration
