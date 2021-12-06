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
        _meshGraph = nullptr;
        _info = {};
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

    for (size_t i = 0; i < 3; i++)
    {
        const MeshGraph::Node *node = _meshGraph->getNode(_info.nodeIndexes[i]);
        _info.nodeLocations[i] = &node->payload.location;
    }

    return true;
}

const MeshIntersectionSearcher::IntersectionInfo &MeshIntersectionSearcher::triangleIntersect(const ray_d &r)
{
    if (_meshGraph == nullptr)
    {
        _info.type = IntersectionInfo::UNINITIALIZED;
        return _info;
    }

    plane_3_corners_d plane;
    for (size_t i = 0; i < 3; i++)
    {
        plane.corner[i] = *_info.nodeLocations[i];
    }

    _info.type = IntersectionInfo::PENDING;
    _info.steps = 0;

    while (true)
    {
        _info.intersectionLocation.fill(NAN);
        if (rayTriangleIntersection(r, plane, _info.intersectionLocation))
        {
            _info.type = IntersectionInfo::INTERSECTION;
            break;
        }

        if (_info.intersectionLocation.hasNaN())
        {
            _info.type = IntersectionInfo::RAY_PARALLEL_TO_PLANE;
            break;
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
            break;
        }

        if (edge->payload.border)
        {
            _info.type = IntersectionInfo::OUTSIDE_BORDER;
            break;
        }

        // switch to the other triangle on that edge
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
            break;
        }

        // reload the changed node
        const MeshGraph::Node *node = _meshGraph->getNode(_info.nodeIndexes[maxIndex]);
        plane.corner[maxIndex] = node->payload.location;
        _info.nodeLocations[maxIndex] = &node->payload.location;
        _info.steps++;
    }

    return _info;
}

} // namespace opencalibration
