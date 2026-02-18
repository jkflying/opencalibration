#include <opencalibration/surface/refine_mesh.hpp>

#include <jk/KDTree.h>
#include <spdlog/spdlog.h>

#include <omp.h>
#include <queue>

namespace opencalibration
{

namespace
{

// Helper to get edge length squared
double edgeLengthSquared(const MeshGraph &mesh, size_t edgeId)
{
    const auto *edge = mesh.getEdge(edgeId);
    if (!edge)
        return 0;

    const auto *srcNode = mesh.getNode(edge->getSource());
    const auto *dstNode = mesh.getNode(edge->getDest());
    if (!srcNode || !dstNode)
        return 0;

    return (srcNode->payload.location - dstNode->payload.location).squaredNorm();
}

// Helper to find edge ID between two nodes (checking both directions)
size_t findEdgeBetween(const MeshGraph &mesh, size_t node1, size_t node2)
{
    // Check both directions since edges are directed
    const auto *node = mesh.getNode(node1);
    if (node)
    {
        for (size_t eid : node->getEdges())
        {
            const auto *e = mesh.getEdge(eid);
            if (e && ((e->getSource() == node1 && e->getDest() == node2) ||
                      (e->getSource() == node2 && e->getDest() == node1)))
            {
                return eid;
            }
        }
    }
    return 0;
}

// Helper to check if a point is inside a triangle (2D, ignoring Z)
bool pointInTriangle2D(double px, double py, const Eigen::Vector3d &v0, const Eigen::Vector3d &v1,
                       const Eigen::Vector3d &v2)
{
    auto sign = [](double p1x, double p1y, double p2x, double p2y, double p3x, double p3y) {
        return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y);
    };

    double d1 = sign(px, py, v0.x(), v0.y(), v1.x(), v1.y());
    double d2 = sign(px, py, v1.x(), v1.y(), v2.x(), v2.y());
    double d3 = sign(px, py, v2.x(), v2.y(), v0.x(), v0.y());

    bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(hasNeg && hasPos);
}

// Find which side of an edge a triangle is on, given one of its vertices (the opposite corner)
int findTriangleSide(const MeshGraph &mesh, size_t edgeId, size_t oppositeVertex)
{
    const auto *edge = mesh.getEdge(edgeId);
    if (!edge)
        return -1;

    if (edge->payload.triangleOppositeNodes[0] == oppositeVertex)
        return 0;
    if (edge->payload.triangleOppositeNodes[1] == oppositeVertex)
        return 1;
    return -1;
}

} // anonymous namespace

std::array<size_t, 3> getTriangleVertices(const MeshGraph &mesh, const TriangleId &tri)
{
    const auto *edge = mesh.getEdge(tri.edgeId);
    if (!edge)
        return {0, 0, 0};

    size_t src = edge->getSource();
    size_t dst = edge->getDest();
    size_t opp = edge->payload.triangleOppositeNodes[tri.side];

    return {src, dst, opp};
}

size_t findLongestEdge(const MeshGraph &mesh, const TriangleId &tri)
{
    auto vertices = getTriangleVertices(mesh, tri);
    if (vertices[0] == 0 && vertices[1] == 0 && vertices[2] == 0)
        return 0;

    // Find edges between all pairs of vertices
    std::array<std::pair<size_t, double>, 3> edges;

    // Edge 0: vertices[0] - vertices[1] (the edge that defines the triangle)
    edges[0] = {tri.edgeId, edgeLengthSquared(mesh, tri.edgeId)};

    // Edge 1: vertices[1] - vertices[2]
    size_t e1 = findEdgeBetween(mesh, vertices[1], vertices[2]);
    edges[1] = {e1, e1 ? edgeLengthSquared(mesh, e1) : 0};

    // Edge 2: vertices[2] - vertices[0]
    size_t e2 = findEdgeBetween(mesh, vertices[2], vertices[0]);
    edges[2] = {e2, e2 ? edgeLengthSquared(mesh, e2) : 0};

    // Find the longest
    size_t longestIdx = 0;
    for (size_t i = 1; i < 3; i++)
    {
        if (edges[i].second > edges[longestIdx].second)
        {
            longestIdx = i;
        }
    }

    return edges[longestIdx].first;
}

TriangleId findTriangleContainingPoint(const MeshGraph &mesh, double x, double y)
{
    // Iterate through all edges and check their triangles
    for (auto it = mesh.cedgebegin(); it != mesh.cedgeend(); ++it)
    {
        const auto &edge = it->second;
        size_t edgeId = it->first;

        // Check side 0
        auto verts = getTriangleVertices(mesh, {edgeId, 0});
        if (verts[0] != 0 || verts[1] != 0 || verts[2] != 0)
        {
            const auto *n0 = mesh.getNode(verts[0]);
            const auto *n1 = mesh.getNode(verts[1]);
            const auto *n2 = mesh.getNode(verts[2]);
            if (n0 && n1 && n2)
            {
                if (pointInTriangle2D(x, y, n0->payload.location, n1->payload.location, n2->payload.location))
                {
                    return {edgeId, 0};
                }
            }
        }

        // Check side 1 if not a border edge
        if (!edge.payload.border)
        {
            verts = getTriangleVertices(mesh, {edgeId, 1});
            if (verts[0] != 0 || verts[1] != 0 || verts[2] != 0)
            {
                const auto *n0 = mesh.getNode(verts[0]);
                const auto *n1 = mesh.getNode(verts[1]);
                const auto *n2 = mesh.getNode(verts[2]);
                if (n0 && n1 && n2)
                {
                    if (pointInTriangle2D(x, y, n0->payload.location, n1->payload.location, n2->payload.location))
                    {
                        return {edgeId, 1};
                    }
                }
            }
        }
    }

    return {0, 0};
}

BisectionResult bisectEdge(MeshGraph &mesh, size_t edgeId)
{
    BisectionResult result{0, 0, {}};

    auto *edge = mesh.getEdge(edgeId);
    if (!edge)
        return result;

    size_t srcId = edge->getSource();
    size_t dstId = edge->getDest();

    const auto *srcNode = mesh.getNode(srcId);
    const auto *dstNode = mesh.getNode(dstId);
    if (!srcNode || !dstNode)
        return result;

    // Create midpoint vertex
    Eigen::Vector3d midpoint = (srcNode->payload.location + dstNode->payload.location) / 2.0;
    size_t midId = mesh.addNode(MeshNode{midpoint});
    result.newVertexId = midId;

    // Get the opposite corners for both triangles
    size_t opp0 = edge->payload.triangleOppositeNodes[0];
    size_t opp1 = edge->payload.triangleOppositeNodes[1];
    bool isBorder = edge->payload.border;

    // Find the other edges of the triangles before we modify anything
    // For triangle 0 (src, dst, opp0): edges are src-opp0 and dst-opp0
    size_t edgeSrcOpp0 = findEdgeBetween(mesh, srcId, opp0);
    size_t edgeDstOpp0 = findEdgeBetween(mesh, dstId, opp0);

    // For triangle 1 (src, dst, opp1): edges are src-opp1 and dst-opp1
    size_t edgeSrcOpp1 = 0, edgeDstOpp1 = 0;
    if (!isBorder && opp1 != 0)
    {
        edgeSrcOpp1 = findEdgeBetween(mesh, srcId, opp1);
        edgeDstOpp1 = findEdgeBetween(mesh, dstId, opp1);
    }

    // Remove the original edge
    mesh.removeEdge(edgeId);

    // Create two new edges to replace the bisected edge
    // src -> mid
    MeshEdge srcMidEdge;
    srcMidEdge.border = isBorder;
    size_t srcMidId = mesh.addEdge(std::move(srcMidEdge), srcId, midId);
    result.splitEdgeIds.push_back(srcMidId);

    // mid -> dst
    MeshEdge midDstEdge;
    midDstEdge.border = isBorder;
    size_t midDstId = mesh.addEdge(std::move(midDstEdge), midId, dstId);
    result.splitEdgeIds.push_back(midDstId);

    // Create edges from midpoint to opposite corners
    // mid -> opp0
    MeshEdge midOpp0Edge;
    midOpp0Edge.border = false;
    size_t midOpp0Id = mesh.addEdge(std::move(midOpp0Edge), midId, opp0);

    // mid -> opp1 (if not border)
    size_t midOpp1Id = 0;
    if (!isBorder && opp1 != 0)
    {
        MeshEdge midOpp1Edge;
        midOpp1Edge.border = false;
        midOpp1Id = mesh.addEdge(std::move(midOpp1Edge), midId, opp1);
    }

    // Now update all the triangle connectivity
    // We now have 4 triangles (or 2 if border):
    // Triangle A: src, mid, opp0
    // Triangle B: mid, dst, opp0
    // Triangle C: src, mid, opp1 (if not border)
    // Triangle D: mid, dst, opp1 (if not border)

    // Update srcMid edge: triangles A (opp0 side) and C (opp1 side)
    auto *srcMidEdgePtr = mesh.getEdge(srcMidId);
    srcMidEdgePtr->payload.triangleOppositeNodes[0] = opp0;
    if (!isBorder && opp1 != 0)
    {
        srcMidEdgePtr->payload.triangleOppositeNodes[1] = opp1;
    }

    // Update midDst edge: triangles B (opp0 side) and D (opp1 side)
    auto *midDstEdgePtr = mesh.getEdge(midDstId);
    midDstEdgePtr->payload.triangleOppositeNodes[0] = opp0;
    if (!isBorder && opp1 != 0)
    {
        midDstEdgePtr->payload.triangleOppositeNodes[1] = opp1;
    }

    // Update midOpp0 edge: triangles A (src side) and B (dst side)
    auto *midOpp0EdgePtr = mesh.getEdge(midOpp0Id);
    midOpp0EdgePtr->payload.triangleOppositeNodes[0] = srcId;
    midOpp0EdgePtr->payload.triangleOppositeNodes[1] = dstId;

    // Update midOpp1 edge: triangles C (src side) and D (dst side)
    if (!isBorder && opp1 != 0 && midOpp1Id != 0)
    {
        auto *midOpp1EdgePtr = mesh.getEdge(midOpp1Id);
        midOpp1EdgePtr->payload.triangleOppositeNodes[0] = srcId;
        midOpp1EdgePtr->payload.triangleOppositeNodes[1] = dstId;
    }

    // Update existing edges that need their opposite corners updated
    // edgeSrcOpp0: was part of (src, dst, opp0), now part of (src, mid, opp0)
    // Its opposite corner that was dst is now mid
    if (edgeSrcOpp0)
    {
        auto *e = mesh.getEdge(edgeSrcOpp0);
        if (e)
        {
            for (int i = 0; i < 2; i++)
            {
                if (e->payload.triangleOppositeNodes[i] == dstId)
                {
                    e->payload.triangleOppositeNodes[i] = midId;
                    break;
                }
            }
        }
    }

    // edgeDstOpp0: was part of (src, dst, opp0), now part of (mid, dst, opp0)
    // Its opposite corner that was src is now mid
    if (edgeDstOpp0)
    {
        auto *e = mesh.getEdge(edgeDstOpp0);
        if (e)
        {
            for (int i = 0; i < 2; i++)
            {
                if (e->payload.triangleOppositeNodes[i] == srcId)
                {
                    e->payload.triangleOppositeNodes[i] = midId;
                    break;
                }
            }
        }
    }

    if (!isBorder && opp1 != 0)
    {
        // edgeSrcOpp1: was part of (src, dst, opp1), now part of (src, mid, opp1)
        if (edgeSrcOpp1)
        {
            auto *e = mesh.getEdge(edgeSrcOpp1);
            if (e)
            {
                for (int i = 0; i < 2; i++)
                {
                    if (e->payload.triangleOppositeNodes[i] == dstId)
                    {
                        e->payload.triangleOppositeNodes[i] = midId;
                        break;
                    }
                }
            }
        }

        // edgeDstOpp1: was part of (src, dst, opp1), now part of (mid, dst, opp1)
        if (edgeDstOpp1)
        {
            auto *e = mesh.getEdge(edgeDstOpp1);
            if (e)
            {
                for (int i = 0; i < 2; i++)
                {
                    if (e->payload.triangleOppositeNodes[i] == srcId)
                    {
                        e->payload.triangleOppositeNodes[i] = midId;
                        break;
                    }
                }
            }
        }
    }

    result.newEdgeId = midOpp0Id;
    return result;
}

size_t refineTriangle(MeshGraph &mesh, const TriangleId &tri, int maxDepth)
{
    if (maxDepth <= 0)
    {
        spdlog::warn("refineTriangle: max depth reached, stopping refinement");
        return 0;
    }

    // Verify triangle exists
    auto vertices = getTriangleVertices(mesh, tri);
    if (vertices[0] == 0 && vertices[1] == 0 && vertices[2] == 0)
    {
        spdlog::debug("refineTriangle: triangle (edge={}, side={}) has invalid vertices", tri.edgeId, tri.side);
        return 0;
    }

    // Find the longest edge (hypotenuse)
    size_t longestEdgeId = findLongestEdge(mesh, tri);
    if (longestEdgeId == 0)
    {
        spdlog::debug("refineTriangle: could not find longest edge for triangle (edge={}, side={})", tri.edgeId,
                      tri.side);
        return 0;
    }

    const auto *longestEdge = mesh.getEdge(longestEdgeId);
    if (!longestEdge)
        return 0;

    // Check if we need to refine the neighbor first
    // This ensures conforming refinement
    size_t trianglesCreated = 0;

    if (!longestEdge->payload.border)
    {
        // Find which side of the longest edge our triangle is on
        int ourSide = -1;
        for (int v = 0; v < 3; v++)
        {
            int side = findTriangleSide(mesh, longestEdgeId, vertices[v]);
            if (side >= 0)
            {
                ourSide = side;
                break;
            }
        }

        if (ourSide >= 0)
        {
            // Get the neighbor triangle
            TriangleId neighbor = {longestEdgeId, 1 - ourSide};

            // Check if neighbor's longest edge is the same as ours
            size_t neighborLongest = findLongestEdge(mesh, neighbor);
            if (neighborLongest != 0 && neighborLongest != longestEdgeId)
            {
                // Neighbor needs to be refined first
                trianglesCreated += refineTriangle(mesh, neighbor, maxDepth - 1);

                // After refining neighbor, our triangle may have changed
                // We need to re-find ourselves and continue
                // The edge we wanted to bisect may no longer exist
                // Try to find a triangle at our center point
                Eigen::Vector3d center = Eigen::Vector3d::Zero();
                for (int i = 0; i < 3; i++)
                {
                    const auto *n = mesh.getNode(vertices[i]);
                    if (n)
                        center += n->payload.location;
                }
                center /= 3.0;

                TriangleId newTri = findTriangleContainingPoint(mesh, center.x(), center.y());
                if (newTri.edgeId != 0)
                {
                    return trianglesCreated + refineTriangle(mesh, newTri, maxDepth - 1);
                }
                return trianglesCreated;
            }
        }
    }

    // Now bisect the longest edge
    bool wasBorder = longestEdge->payload.border;
    auto result = bisectEdge(mesh, longestEdgeId);
    if (result.newVertexId != 0)
    {
        // We created 2 new triangles on each side (4 total, or 2 if border)
        trianglesCreated += wasBorder ? 2 : 4;
    }

    return trianglesCreated;
}

size_t refineAtPoint(MeshGraph &mesh, double x, double y, int levels)
{
    size_t totalCreated = 0;

    for (int level = 0; level < levels; level++)
    {
        TriangleId tri = findTriangleContainingPoint(mesh, x, y);
        if (tri.edgeId == 0)
        {
            spdlog::debug("refineAtPoint: no triangle found at ({}, {})", x, y);
            break;
        }

        size_t created = refineTriangle(mesh, tri);
        if (created == 0)
            break;

        totalCreated += created;
    }

    return totalCreated;
}

size_t refineWhere(MeshGraph &mesh, std::function<bool(double x, double y, double z)> shouldRefine, int maxIterations)
{
    size_t totalCreated = 0;

    for (int iter = 0; iter < maxIterations; iter++)
    {
        // Collect triangles to refine (using a set to avoid duplicates)
        ankerl::unordered_dense::set<size_t> visitedEdges;
        std::vector<TriangleId> toRefine;

        for (auto it = mesh.cedgebegin(); it != mesh.cedgeend(); ++it)
        {
            size_t edgeId = it->first;
            if (visitedEdges.count(edgeId))
                continue;
            visitedEdges.insert(edgeId);

            const auto &edge = it->second;

            // Check side 0
            {
                TriangleId tri{edgeId, 0};
                auto verts = getTriangleVertices(mesh, tri);
                if (verts[0] != 0 || verts[1] != 0 || verts[2] != 0)
                {
                    Eigen::Vector3d center = Eigen::Vector3d::Zero();
                    int validNodes = 0;
                    for (int i = 0; i < 3; i++)
                    {
                        const auto *n = mesh.getNode(verts[i]);
                        if (n)
                        {
                            center += n->payload.location;
                            validNodes++;
                        }
                    }
                    if (validNodes == 3)
                    {
                        center /= 3.0;
                        if (shouldRefine(center.x(), center.y(), center.z()))
                        {
                            toRefine.push_back(tri);
                        }
                    }
                }
            }

            // Check side 1 if not border
            if (!edge.payload.border)
            {
                TriangleId tri{edgeId, 1};
                auto verts = getTriangleVertices(mesh, tri);
                if (verts[0] != 0 || verts[1] != 0 || verts[2] != 0)
                {
                    Eigen::Vector3d center = Eigen::Vector3d::Zero();
                    int validNodes = 0;
                    for (int i = 0; i < 3; i++)
                    {
                        const auto *n = mesh.getNode(verts[i]);
                        if (n)
                        {
                            center += n->payload.location;
                            validNodes++;
                        }
                    }
                    if (validNodes == 3)
                    {
                        center /= 3.0;
                        if (shouldRefine(center.x(), center.y(), center.z()))
                        {
                            toRefine.push_back(tri);
                        }
                    }
                }
            }
        }

        if (toRefine.empty())
            break;

        size_t createdThisIter = 0;
        for (const auto &tri : toRefine)
        {
            // Triangle may have been invalidated by previous refinement
            auto verts = getTriangleVertices(mesh, tri);
            if (verts[0] == 0 && verts[1] == 0 && verts[2] == 0)
                continue;

            createdThisIter += refineTriangle(mesh, tri);
        }

        if (createdThisIter == 0)
            break;

        totalCreated += createdThisIter;
    }

    return totalCreated;
}

TriangleLocator::TriangleLocator(const MeshGraph &m) : _mesh(m)
{
    for (auto it = _mesh.cedgebegin(); it != _mesh.cedgeend(); ++it)
    {
        size_t edgeId = it->first;
        const auto &edge = it->second;

        for (int side = 0; side < 2; side++)
        {
            if (side == 1 && edge.payload.border)
                continue;

            TriangleId tri{edgeId, side};
            auto verts = getTriangleVertices(_mesh, tri);
            if (verts[0] == 0 && verts[1] == 0 && verts[2] == 0)
                continue;

            const auto *n0 = _mesh.getNode(verts[0]);
            const auto *n1 = _mesh.getNode(verts[1]);
            const auto *n2 = _mesh.getNode(verts[2]);
            if (!n0 || !n1 || !n2)
                continue;

            double cx = (n0->payload.location.x() + n1->payload.location.x() + n2->payload.location.x()) / 3.0;
            double cy = (n0->payload.location.y() + n1->payload.location.y() + n2->payload.location.y()) / 3.0;

            _centroidTree.addPoint({cx, cy}, tri, false);
        }
    }
    _centroidTree.splitOutstanding();
}

TriangleId TriangleLocator::find(double x, double y) const
{
    if (_centroidTree.size() == 0)
        return {0, 0};

    auto nearest = _centroidTree.search({x, y});
    TriangleId current = nearest.payload;

    for (int step = 0; step < 100; step++)
    {
        auto verts = getTriangleVertices(_mesh, current);
        if (verts[0] == 0 && verts[1] == 0 && verts[2] == 0)
            break;

        const auto *n0 = _mesh.getNode(verts[0]);
        const auto *n1 = _mesh.getNode(verts[1]);
        const auto *n2 = _mesh.getNode(verts[2]);
        if (!n0 || !n1 || !n2)
            break;

        const auto &p0 = n0->payload.location;
        const auto &p1 = n1->payload.location;
        const auto &p2 = n2->payload.location;

        // Half-plane sign tests
        auto sign = [](double px, double py, double ax, double ay, double bx, double by) {
            return (px - bx) * (ay - by) - (ax - bx) * (py - by);
        };

        double d0 = sign(x, y, p0.x(), p0.y(), p1.x(), p1.y()); // edge v0-v1
        double d1 = sign(x, y, p1.x(), p1.y(), p2.x(), p2.y()); // edge v1-v2
        double d2 = sign(x, y, p2.x(), p2.y(), p0.x(), p0.y()); // edge v2-v0

        bool hasNeg = (d0 < 0) || (d1 < 0) || (d2 < 0);
        bool hasPos = (d0 > 0) || (d1 > 0) || (d2 > 0);

        if (!(hasNeg && hasPos))
            return current; // Point is inside this triangle

        // Determine expected sign (majority vote) and pick the most-violated edge
        int negCount = (d0 < 0) + (d1 < 0) + (d2 < 0);
        bool expectPositive = negCount < 2;

        double worstVal = 0;
        int worstEdge = -1;

        auto checkEdge = [&](int edgeIdx, double d) {
            if ((d > 0) != expectPositive && std::abs(d) > worstVal)
            {
                worstVal = std::abs(d);
                worstEdge = edgeIdx;
            }
        };
        checkEdge(0, d0);
        checkEdge(1, d1);
        checkEdge(2, d2);

        if (worstEdge < 0)
            break;

        // Cross the worst edge to the neighboring triangle
        // Edge 0: verts[0]-verts[1] = the defining edge of current TriangleId
        // Edge 1: verts[1]-verts[2]
        // Edge 2: verts[2]-verts[0]
        TriangleId neighbor{0, 0};

        if (worstEdge == 0)
        {
            // Cross the defining edge to the other side
            const auto *edge = _mesh.getEdge(current.edgeId);
            if (edge && !edge->payload.border)
            {
                neighbor = {current.edgeId, 1 - current.side};
            }
        }
        else
        {
            size_t va = verts[worstEdge];
            size_t vb = verts[(worstEdge + 1) % 3];
            size_t crossEdgeId = findEdgeBetween(_mesh, va, vb);
            if (crossEdgeId != 0)
            {
                const auto *crossEdge = _mesh.getEdge(crossEdgeId);
                if (crossEdge && !crossEdge->payload.border)
                {
                    size_t oppositeVertex = verts[(worstEdge + 2) % 3];
                    int currentSide = findTriangleSide(_mesh, crossEdgeId, oppositeVertex);
                    if (currentSide >= 0)
                    {
                        neighbor = {crossEdgeId, 1 - currentSide};
                    }
                }
            }
        }

        if (neighbor.edgeId == 0)
            break; // Hit border or invalid edge

        current = neighbor;
    }

    // Fallback: brute force search (should rarely be needed)
    return findTriangleContainingPoint(_mesh, x, y);
}

ankerl::unordered_dense::map<TriangleId, TrianglePointStats, TriangleIdHash> countPointsPerTriangle(
    const MeshGraph &mesh, const std::vector<point_cloud> &points)
{
    struct Accumulator
    {
        size_t count = 0;
        double sumDist = 0;
        double sumDistSq = 0;
    };

    ankerl::unordered_dense::map<TriangleId, Accumulator, TriangleIdHash> accumulators;

    struct TrianglePlane
    {
        Eigen::Vector3d normal;
        Eigen::Vector3d origin;
    };
    ankerl::unordered_dense::map<TriangleId, TrianglePlane, TriangleIdHash> planeCache;

    spdlog::debug("countPointsPerTriangle: mesh has {} nodes, {} edges", mesh.size_nodes(), mesh.size_edges());

    TriangleLocator locator(mesh);

    for (const auto &cloud : points)
    {
        for (const auto &p : cloud)
        {
            TriangleId tri = locator.find(p.x(), p.y());
            if (tri.edgeId == 0)
                continue;

            auto &acc = accumulators[tri];
            acc.count++;

            auto planeIt = planeCache.find(tri);
            if (planeIt == planeCache.end())
            {
                auto verts = getTriangleVertices(mesh, tri);
                const auto *n0 = mesh.getNode(verts[0]);
                const auto *n1 = mesh.getNode(verts[1]);
                const auto *n2 = mesh.getNode(verts[2]);
                if (n0 && n1 && n2)
                {
                    Eigen::Vector3d normal = (n1->payload.location - n0->payload.location)
                                                 .cross(n2->payload.location - n0->payload.location)
                                                 .normalized();
                    planeIt = planeCache.emplace(tri, TrianglePlane{normal, n0->payload.location}).first;
                }
            }

            if (planeIt != planeCache.end())
            {
                double dist = (p - planeIt->second.origin).dot(planeIt->second.normal);
                acc.sumDist += dist;
                acc.sumDistSq += dist * dist;
            }
        }
    }

    ankerl::unordered_dense::map<TriangleId, TrianglePointStats, TriangleIdHash> result;
    for (const auto &[tri, acc] : accumulators)
    {
        TrianglePointStats stats;
        stats.count = acc.count;
        if (acc.count > 1)
        {
            double mean = acc.sumDist / acc.count;
            stats.distanceVariance = acc.sumDistSq / acc.count - mean * mean;
        }
        result[tri] = stats;
    }

    spdlog::debug("countPointsPerTriangle: found {} unique triangles with points", result.size());

    return result;
}

size_t refineByPointDensity(MeshGraph &mesh, const std::vector<point_cloud> &points, size_t maxPointsPerTriangle,
                            double minDistanceVariance, int maxIterations)
{
    size_t totalCreated = 0;

    for (int iter = 0; iter < maxIterations; iter++)
    {
        auto stats = countPointsPerTriangle(mesh, points);

        std::vector<TriangleId> toRefine;
        for (const auto &[tri, s] : stats)
        {
            if (s.count > maxPointsPerTriangle && s.distanceVariance > minDistanceVariance)
            {
                toRefine.push_back(tri);
                spdlog::debug("refineByPointDensity: triangle (edge={}, side={}) has {} points, variance {}",
                              tri.edgeId, tri.side, s.count, s.distanceVariance);
            }
        }

        if (toRefine.empty())
        {
            spdlog::info("refineByPointDensity: converged after {} iterations, {} triangles created", iter,
                         totalCreated);
            break;
        }

        spdlog::info("refineByPointDensity: iteration {}, refining {} triangles exceeding {} points", iter,
                     toRefine.size(), maxPointsPerTriangle);

        size_t createdThisIter = 0;
        for (const auto &tri : toRefine)
        {
            // Triangle may have been invalidated by previous refinement
            auto verts = getTriangleVertices(mesh, tri);
            if (verts[0] == 0 && verts[1] == 0 && verts[2] == 0)
            {
                spdlog::debug("refineByPointDensity: skipping invalidated triangle (edge={}, side={})", tri.edgeId,
                              tri.side);
                continue;
            }

            size_t created = refineTriangle(mesh, tri);
            spdlog::debug("refineByPointDensity: refineTriangle returned {} for (edge={}, side={})", created,
                          tri.edgeId, tri.side);
            createdThisIter += created;
        }

        if (createdThisIter == 0)
        {
            spdlog::info("refineByPointDensity: no triangles created in iteration {}, stopping", iter);
            break;
        }

        totalCreated += createdThisIter;
    }

    return totalCreated;
}

void refineMesh(const MeasurementGraph & /*measurementGraph*/, MeshGraph & /*meshGraph*/)
{
    // Legacy function - use refineByPointDensity directly with the point clouds from surface_model
}

surface_model mergeSurfaceModels(const std::vector<surface_model> &surfaces)
{
    if (surfaces.empty())
    {
        return surface_model{};
    }

    if (surfaces.size() == 1)
    {
        return surfaces[0];
    }

    // Use the first surface's mesh as the base structure
    surface_model result;
    result.mesh = surfaces[0].mesh;

    // For each surface, count points per vertex (sum of points in adjacent triangles)
    // vertex_id -> (weighted_position_sum, total_weight)
    ankerl::unordered_dense::map<size_t, std::pair<Eigen::Vector3d, double>> vertexWeights;

    // Initialize with zero weights
    for (auto it = result.mesh.cnodebegin(); it != result.mesh.cnodeend(); ++it)
    {
        vertexWeights[it->first] = {Eigen::Vector3d::Zero(), 0.0};
    }

    std::vector<ankerl::unordered_dense::map<size_t, std::pair<Eigen::Vector3d, double>>> threadLocalWeights(surfaces.size());

#pragma omp parallel for schedule(dynamic)
    for (size_t surfIdx = 0; surfIdx < surfaces.size(); surfIdx++)
    {
        const auto &surf = surfaces[surfIdx];

        if (surf.mesh.size_nodes() == 0)
        {
            continue;
        }

        // Count points per triangle for this surface
        auto triangleCounts = countPointsPerTriangle(surf.mesh, surf.cloud);

        // For each vertex, accumulate weights from adjacent triangles
        ankerl::unordered_dense::map<size_t, size_t> vertexPointCounts;

        for (const auto &[tri, triStats] : triangleCounts)
        {
            auto verts = getTriangleVertices(surf.mesh, tri);
            if (verts[0] == 0 && verts[1] == 0 && verts[2] == 0)
            {
                continue;
            }

            // Add this triangle's point count to each of its vertices
            for (int i = 0; i < 3; i++)
            {
                vertexPointCounts[verts[i]] += triStats.count;
            }
        }

        auto &threadLocal = threadLocalWeights[surfIdx];
        for (const auto &[nodeId, pointCount] : vertexPointCounts)
        {
            auto nodePtr = surf.mesh.getNode(nodeId);
            if (!nodePtr)
                continue;

            const Eigen::Vector3d &pos = nodePtr->payload.location;
            double weight = static_cast<double>(pointCount);
            threadLocal[nodeId] = {pos * weight, weight};
        }

#pragma omp critical(merge_clouds)
        {
            for (const auto &cloud : surf.cloud)
            {
                result.cloud.push_back(cloud);
            }
        }
    }

    for (size_t surfIdx = 0; surfIdx < surfaces.size(); surfIdx++)
    {
        for (const auto &[nodeId, weightPair] : threadLocalWeights[surfIdx])
        {
            auto &[sumPos, sumWeight] = vertexWeights[nodeId];
            sumPos += weightPair.first;
            sumWeight += weightPair.second;
        }
    }

    // Apply weighted average positions to result mesh
    for (auto nodeIt = result.mesh.nodebegin(); nodeIt != result.mesh.nodeend(); ++nodeIt)
    {
        size_t nodeId = nodeIt->first;
        auto &[sumPos, sumWeight] = vertexWeights[nodeId];

        if (sumWeight > 0)
        {
            nodeIt->second.payload.location = sumPos / sumWeight;
        }
        // If no weight (no points nearby), keep original position from first surface
    }

    spdlog::info("Merged {} surface models into one with {} nodes, {} edges, {} point clouds", surfaces.size(),
                 result.mesh.size_nodes(), result.mesh.size_edges(), result.cloud.size());

    return result;
}

} // namespace opencalibration
