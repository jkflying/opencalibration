#pragma once

#include <opencalibration/types/graph.hpp>

#include <eigen3/Eigen/Core>

namespace opencalibration
{

// used for internal implementation only
struct MeshNode
{

    Eigen::Vector3d location;
};

struct MeshEdge
{
    bool border = false;
    std::array<size_t, 2> triangleOppositeNodes{};
};

using MeshGraph = DirectedGraph<MeshNode, MeshEdge>;
} // namespace opencalibration
