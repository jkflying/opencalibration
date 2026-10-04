#pragma once

#include <opencalibration/types/mesh_graph.hpp>
#include <opencalibration/types/plane.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <ankerl/unordered_dense.h>

namespace opencalibration
{
struct surface_model
{
    std::vector<point_cloud> cloud;
    MeshGraph mesh;
    ankerl::unordered_dense::set<size_t> observed_vertices{};
};
} // namespace opencalibration
