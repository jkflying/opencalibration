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
        IntersectionInfo()
            : type(PENDING), nodeIndexes({}), nodeLocations({}), intersectionLocation(NAN, NAN, NAN), steps(0)
        {
        }
        enum INTERSECTION_TYPE
        {
            UNINITIALIZED,
            PENDING,
            INTERSECTION,
            OUTSIDE_BORDER,
            RAY_PARALLEL_TO_PLANE,
            GRAPH_STRUCTURE_INCONSISTENT

        } type;
        std::array<size_t, 3> nodeIndexes;
        std::array<const Eigen::Vector3d *, 3> nodeLocations;
        Eigen::Vector3d intersectionLocation;
        size_t steps;
    };

    bool init(const MeshGraph &meshGraph, const IntersectionInfo &info = {});

    bool reinit();

    // Faster if called consecutively with rays that intersect the mesh near to each other
    // Note: not threadsafe, use one instance of MeshIntersectionSearcher per thread
    const IntersectionInfo &triangleIntersect(const ray_d &r);
    const IntersectionInfo &lastResult();

    bool initialized() const
    {
        return _meshGraph != nullptr;
    }

  private:
    const MeshGraph *_meshGraph;
    IntersectionInfo _info;

    std::vector<size_t> _keepNodes;
};
} // namespace opencalibration
