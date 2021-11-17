#pragma once

#include <opencalibration/types/graph.hpp>

#include <eigen3/Eigen/Core>

namespace opencalibration
{

// used for internal implementation only
struct MeshNode
{
    Eigen::Vector3d location;

    bool operator==(const MeshNode &other) const
    {
        return location == other.location;
    }
};

struct MeshEdge
{
    bool border = false;
    std::array<size_t, 2> triangleOppositeNodes{};

    bool operator==(const MeshEdge &other) const
    {
        return border == other.border && triangleOppositeNodes == other.triangleOppositeNodes;
    }
};

using MeshGraph = DirectedGraph<MeshNode, MeshEdge>;
} // namespace opencalibration
