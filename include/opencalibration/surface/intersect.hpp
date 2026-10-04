#pragma once

#include <opencalibration/types/mesh_graph.hpp>
#include <opencalibration/types/ray.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <memory>

namespace opencalibration
{

class MeshIntersectionSearcher
{
  public:
    struct IntersectionInfo
    {
        IntersectionInfo() : intersectionLocation(NAN, NAN, NAN)
        {
        }
        enum INTERSECTION_TYPE
        {
            UNINITIALIZED,
            PENDING,
            INTERSECTION,
            OUTSIDE_BORDER,
            RAY_PARALLEL_TO_PLANE,
            GRAPH_STRUCTURE_INCONSISTENT,
            MAX_STEPS_EXCEEDED

        } type = PENDING;
        std::array<size_t, 3> nodeIndexes = {};
        std::array<const Eigen::Vector3d *, 3> nodeLocations = {};
        Eigen::Vector3d intersectionLocation;
        size_t steps = 0;
    };

    [[nodiscard]] bool init(const MeshGraph &meshGraph, const IntersectionInfo &info = {});

    [[nodiscard]] bool reinit();

    // Faster if called consecutively with rays that intersect the mesh near to each other
    // Note: not threadsafe, use one instance of MeshIntersectionSearcher per thread
    const IntersectionInfo &triangleIntersect(const ray_d &r);
    const IntersectionInfo &lastResult();

    [[nodiscard]] bool initialized() const
    {
        return _meshGraph != nullptr;
    }

  private:
    const MeshGraph *_meshGraph = nullptr;
    IntersectionInfo _info;
    IntersectionInfo _lastIntersection;

    std::vector<size_t> _keepNodes;
};

class MeshLineOfSight
{
  public:
    [[nodiscard]] bool init(const MeshGraph &meshGraph);

    double surfaceHeight(const Eigen::Vector2d &xy);

    bool surfaceVisibleFrom(const Eigen::Vector2d &xy, const Eigen::Vector3d &viewpoint);

  private:
    struct MaxMipmap;
    struct ShadowRay;

    bool traceUnoccluded(const ShadowRay &ray, double tMax, double epsilon);
    bool marchUnoccluded(const ShadowRay &ray, double tStart, double tEnd, double epsilon);

    MeshIntersectionSearcher _searcher;
    std::shared_ptr<const MaxMipmap> _maxMipmap;
    double _maxSurfaceZ = NAN;
    double _marchStep = NAN;
};

std::vector<MeshLineOfSight> sightlinesOver(const std::vector<surface_model> &surfaces);

bool surfaceVisibleFrom(std::vector<MeshLineOfSight> &sightlines, const Eigen::Vector3d &point,
                        const Eigen::Vector3d &viewpoint);
} // namespace opencalibration
