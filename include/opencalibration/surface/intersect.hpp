#pragma once

#include <opencalibration/types/mesh_graph.hpp>
#include <opencalibration/types/ray.hpp>

namespace opencalibration
{

class MeshIntersectionSearcher
{
  public:
    struct IntersectionInfo
    {
        enum INTERSECTION_TYPE
        {
            INTERSECTION,
            OUTSIDE_BORDER,
            RAY_PARALLEL_TO_PLANE,
            GRAPH_STRUCTURE_INCONSISTENT

        } type;
        std::array<size_t, 3> nodeIndexes;
        std::array<Eigen::Vector3d, 3> nodeLocations;
        Eigen::Vector3d intersectionLocation;
    };

    bool init(const MeshGraph &meshGraph, const IntersectionInfo &info = {});

    // Faster if called consecutively with rays that intersect the mesh near to each other
    const IntersectionInfo &triangleIntersect(const ray_d &r);

  private:
    const MeshGraph *_meshGraph;
    IntersectionInfo _info;

    std::vector<size_t> keepNodes;
};
} // namespace opencalibration
