#include <opencalibration/surface/intersect.hpp>

#include <opencalibration/geometry/intersection.hpp>
#include <opencalibration/geometry/utils.hpp>

#include <spdlog/spdlog.h>

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
        if (node == nullptr)
        {
            return false;
        }
        _info.nodeLocations[i] = &node->payload.location;
    }

    return true;
}

bool MeshIntersectionSearcher::reinit()
{
    return init(*_meshGraph);
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
        if (anticlockwise(plane.corner))
        {
            std::swap(plane.corner[0], plane.corner[1]);
            std::swap(_info.nodeIndexes[0], _info.nodeIndexes[1]);
            std::swap(_info.nodeLocations[0], _info.nodeLocations[1]);
        }

        _info.intersectionLocation.fill(NAN);
        if (!rayPlaneIntersection(r, cornerPlane2normOffsetPlane(plane), _info.intersectionLocation) ||
            _info.intersectionLocation.hasNaN())
        {
            _info.type = IntersectionInfo::RAY_PARALLEL_TO_PLANE;
            break;
        }

        // find an edge which puts two points of the old triangle + test point in the opposite rotation order
        size_t edgeIndex = 3;
        for (int i = 0; i < 3; i++)
        {
            const std::array<Eigen::Vector3d, 3> corners{_info.intersectionLocation, plane.corner[i],
                                                         plane.corner[(i + 1) % 3]};
            if (anticlockwise(corners))
            {
                edgeIndex = i;
                break;
            }
        }

        // the point is inside the triangle because the rotation stayed the same for all point combos
        if (edgeIndex == 3)
        {
            _info.type = IntersectionInfo::INTERSECTION;
            break;
        }

        _keepNodes = {_info.nodeIndexes[edgeIndex], _info.nodeIndexes[(edgeIndex + 1) % 3]};

        const MeshGraph::Edge *edge = _meshGraph->getEdge(_keepNodes[0], _keepNodes[1]);
        if (edge == nullptr)
        {
            edge = _meshGraph->getEdge(_keepNodes[1], _keepNodes[0]);
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

        size_t replacedNode = (edgeIndex + 2) % 3;

        // switch to the other triangle on that edge
        if (edge->payload.triangleOppositeNodes[0] == _info.nodeIndexes[replacedNode])
        {
            _info.nodeIndexes[replacedNode] = edge->payload.triangleOppositeNodes[1];
        }
        else if (edge->payload.triangleOppositeNodes[1] == _info.nodeIndexes[replacedNode])
        {
            _info.nodeIndexes[replacedNode] = edge->payload.triangleOppositeNodes[0];
        }
        else
        {
            _info.type = IntersectionInfo::GRAPH_STRUCTURE_INCONSISTENT;
            break;
        }

        // reload the changed node
        const MeshGraph::Node *node = _meshGraph->getNode(_info.nodeIndexes[replacedNode]);
        plane.corner[replacedNode] = node->payload.location;
        _info.nodeLocations[replacedNode] = &node->payload.location;
        _info.steps++;

        if (_info.steps > _meshGraph->size_edges())
        {
            spdlog::error("Graph intersection search failed after {} steps!", _info.steps);
            _info.type = IntersectionInfo::GRAPH_STRUCTURE_INCONSISTENT;
            break;
        }
    }

    return _info;
}

const MeshIntersectionSearcher::IntersectionInfo &MeshIntersectionSearcher::lastResult()
{
    return _info;
}

} // namespace opencalibration
