#include <opencalibration/surface/intersect.hpp>

#include <opencalibration/geometry/intersection.hpp>

namespace opencalibration
{
bool MeshIntersectionSearcher::init(const MeshGraph &meshGraph, const IntersectionInfo &info)
{
    _meshGraph = &meshGraph;
    _info = info;

    if (_meshGraph->size_nodes() == 0 || _meshGraph->size_edges() == 0)
    {
        return false;
    }

    // check if triangle is valid and can be used as a starting point for the search
    bool valid = true;
    for (size_t i = 0; i < 3; i++)
    {
        valid &= _info.nodeIndexes[i] != _info.nodeIndexes[(i + 1) % 3];
        valid &= _meshGraph->getNode(_info.nodeIndexes[i]) != nullptr;
    }

    if (!valid) // populate with a random edge as a starting point
    {
        const auto &edge = _meshGraph->cedgebegin()->second;
        _info.nodeIndexes[0] = edge.getSource();
        _info.nodeIndexes[1] = edge.getDest();
        _info.nodeIndexes[2] = edge.payload.triangleOppositeNodes[0];
    }

    return true;
}

const MeshIntersectionSearcher::IntersectionInfo &MeshIntersectionSearcher::triangleIntersect(const ray_d &r)
{

    plane_3_corners_d plane;
    for (size_t i = 0; i < 3; i++)
    {
        const MeshGraph::Node *node = _meshGraph->getNode(_info.nodeIndexes[i]);
        plane.corner[i] = node->payload.location;
    }

    while (true)
    {
        _info.intersectionLocation.fill(NAN);
        if (rayTriangleIntersection(r, plane, _info.intersectionLocation))
        {
            return _info;
        }

        if (_info.intersectionLocation.hasNaN())
        {
            _info.type = IntersectionInfo::RAY_PARALLEL_TO_PLANE;
            return _info; // can't continue due to numerical issues, ray and plane parallel
        }

        // otherwise find the edge across from the furthest vertex to the intersection
        double maxDist = 0;
        size_t maxIndex = 0;
        for (size_t i = 0; i < 3; i++)
        {
            const double distSq = (plane.corner[i] - _info.intersectionLocation).squaredNorm();
            if (distSq > maxDist)
            {
                maxDist = distSq;
                maxIndex = i;
            }
        }

        keepNodes = {_info.nodeIndexes[0], _info.nodeIndexes[1], _info.nodeIndexes[2]};
        keepNodes.erase(keepNodes.begin() + maxIndex);

        const MeshGraph::Edge *edge = _meshGraph->getEdge(keepNodes[0], keepNodes[1]);
        if (edge == nullptr)
        {
            edge = _meshGraph->getEdge(keepNodes[1], keepNodes[0]);
        }

        if (edge == nullptr)
        {
            _info.type = IntersectionInfo::GRAPH_STRUCTURE_INCONSISTENT;
            return _info; // internal graph structure messed up
        }

        if (edge->payload.border)
        {
            _info.type = IntersectionInfo::OUTSIDE_BORDER;
            return _info; // intersection point is off the mesh
        }

        // switch to the other triangle on that edge, and try again
        if (edge->payload.triangleOppositeNodes[0] == _info.nodeIndexes[maxIndex])
        {
            _info.nodeIndexes[maxIndex] = edge->payload.triangleOppositeNodes[1];
        }
        else if (edge->payload.triangleOppositeNodes[1] == _info.nodeIndexes[maxIndex])
        {
            _info.nodeIndexes[maxIndex] = edge->payload.triangleOppositeNodes[0];
        }
        else
        {
            _info.type = IntersectionInfo::GRAPH_STRUCTURE_INCONSISTENT;
            return _info; // we weren't in this triangle somehow
        }

        const MeshGraph::Node *node = _meshGraph->getNode(_info.nodeIndexes[maxIndex]);
        plane.corner[maxIndex] = node->payload.location;
    }
}

} // namespace opencalibration
