#include <opencalibration/surface/refine_mesh.hpp>

#include <jk/KDTree.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <omp.h>
#include <optional>
#include <queue>

namespace opencalibration
{

namespace
{

double edgeLengthSquared(const MeshGraph &mesh, size_t edgeId)
{
    const auto *edge = mesh.getEdge(edgeId);
    if (!edge)
        return 0;

    const auto *srcNode = mesh.getNode(edge->getSource());
    const auto *dstNode = mesh.getNode(edge->getDest());
    if (!srcNode || !dstNode)
        return 0;

    // 2D metric: rest of the pipeline (point-in-triangle, locator, minTriangleSizeMeters) is XY.
    return (srcNode->payload.location.head<2>() - dstNode->payload.location.head<2>()).squaredNorm();
}

size_t findEdgeBetween(const MeshGraph &mesh, size_t node1, size_t node2)
{
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

// Search for a triangle containing (x,y) among triangles adjacent to any of the given vertices.
// O(degree) instead of O(E) — original vertices survive bisection so the centroid of a split
// triangle always lands in a sub-triangle incident to one of the original vertices.
TriangleId findTriangleNearVertices(const MeshGraph &mesh, const std::array<size_t, 3> &vertices, double x, double y)
{
    for (size_t vtxId : vertices)
    {
        const auto *vtxNode = mesh.getNode(vtxId);
        if (!vtxNode)
            continue;
        for (size_t eid : vtxNode->getEdges())
        {
            for (int side = 0; side < 2; side++)
            {
                TriangleId candidate{eid, side};
                auto cverts = getTriangleVertices(mesh, candidate);
                if (cverts[0] == 0 && cverts[1] == 0 && cverts[2] == 0)
                    continue;
                const auto *cn0 = mesh.getNode(cverts[0]);
                const auto *cn1 = mesh.getNode(cverts[1]);
                const auto *cn2 = mesh.getNode(cverts[2]);
                if (!cn0 || !cn1 || !cn2)
                    continue;
                if (pointInTriangle2D(x, y, cn0->payload.location, cn1->payload.location, cn2->payload.location))
                    return candidate;
            }
        }
    }
    return {0, 0};
}

TriangleId lowestEdgeTriangleId(const MeshGraph &mesh, const TriangleId &tri)
{
    const auto verts = getTriangleVertices(mesh, tri);
    TriangleId lowest = tri;
    for (size_t i = 1; i < 3; i++)
    {
        const size_t edgeId = findEdgeBetween(mesh, verts[i], verts[(i + 1) % 3]);
        const int side = findTriangleSide(mesh, edgeId, verts[(i + 2) % 3]);
        if (edgeId != 0 && side >= 0 && edgeId < lowest.edgeId)
        {
            lowest = {edgeId, side};
        }
    }
    return lowest;
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

    std::array<std::pair<size_t, double>, 3> edges;
    edges[0] = {tri.edgeId, edgeLengthSquared(mesh, tri.edgeId)};

    size_t e1 = findEdgeBetween(mesh, vertices[1], vertices[2]);
    edges[1] = {e1, e1 ? edgeLengthSquared(mesh, e1) : 0};

    size_t e2 = findEdgeBetween(mesh, vertices[2], vertices[0]);
    edges[2] = {e2, e2 ? edgeLengthSquared(mesh, e2) : 0};

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
    for (auto it = mesh.cedgebegin(); it != mesh.cedgeend(); ++it)
    {
        const auto &edge = it->second;
        size_t edgeId = it->first;

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

    Eigen::Vector3d midpoint = (srcNode->payload.location + dstNode->payload.location) / 2.0;
    size_t midId = mesh.addNode(MeshNode{midpoint});
    result.newVertexId = midId;

    size_t opp0 = edge->payload.triangleOppositeNodes[0];
    size_t opp1 = edge->payload.triangleOppositeNodes[1];
    bool isBorder = edge->payload.border;

    // Capture adjacent edges before removing the bisected edge
    size_t edgeSrcOpp0 = findEdgeBetween(mesh, srcId, opp0);
    size_t edgeDstOpp0 = findEdgeBetween(mesh, dstId, opp0);

    size_t edgeSrcOpp1 = 0, edgeDstOpp1 = 0;
    if (!isBorder && opp1 != 0)
    {
        edgeSrcOpp1 = findEdgeBetween(mesh, srcId, opp1);
        edgeDstOpp1 = findEdgeBetween(mesh, dstId, opp1);
    }

    mesh.removeEdge(edgeId);

    MeshEdge srcMidEdge;
    srcMidEdge.border = isBorder;
    size_t srcMidId = mesh.addEdge(srcMidEdge, srcId, midId);
    result.splitEdgeIds.push_back(srcMidId);

    MeshEdge midDstEdge;
    midDstEdge.border = isBorder;
    size_t midDstId = mesh.addEdge(midDstEdge, midId, dstId);
    result.splitEdgeIds.push_back(midDstId);

    MeshEdge midOpp0Edge;
    midOpp0Edge.border = false;
    size_t midOpp0Id = mesh.addEdge(midOpp0Edge, midId, opp0);

    size_t midOpp1Id = 0;
    if (!isBorder && opp1 != 0)
    {
        MeshEdge midOpp1Edge;
        midOpp1Edge.border = false;
        midOpp1Id = mesh.addEdge(midOpp1Edge, midId, opp1);
    }

    // 4 triangles after bisection (2 if border):
    // Triangle A: src, mid, opp0
    // Triangle B: mid, dst, opp0
    // Triangle C: src, mid, opp1 (if not border)
    // Triangle D: mid, dst, opp1 (if not border)
    auto *srcMidEdgePtr = mesh.getEdge(srcMidId);
    srcMidEdgePtr->payload.triangleOppositeNodes[0] = opp0;
    if (!isBorder && opp1 != 0)
    {
        srcMidEdgePtr->payload.triangleOppositeNodes[1] = opp1;
    }

    auto *midDstEdgePtr = mesh.getEdge(midDstId);
    midDstEdgePtr->payload.triangleOppositeNodes[0] = opp0;
    if (!isBorder && opp1 != 0)
    {
        midDstEdgePtr->payload.triangleOppositeNodes[1] = opp1;
    }

    auto *midOpp0EdgePtr = mesh.getEdge(midOpp0Id);
    midOpp0EdgePtr->payload.triangleOppositeNodes[0] = srcId;
    midOpp0EdgePtr->payload.triangleOppositeNodes[1] = dstId;

    if (!isBorder && opp1 != 0 && midOpp1Id != 0)
    {
        auto *midOpp1EdgePtr = mesh.getEdge(midOpp1Id);
        midOpp1EdgePtr->payload.triangleOppositeNodes[0] = srcId;
        midOpp1EdgePtr->payload.triangleOppositeNodes[1] = dstId;
    }

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
    size_t trianglesCreated = 0;
    TriangleId currentTri = tri;

    // Loop handles retries after neighbor refinement invalidates our triangle.
    // Depth is only consumed by neighbor propagation, not by retries.
    for (;;)
    {
        if (maxDepth <= 0)
        {
            spdlog::warn("refineTriangle: max depth reached, stopping refinement");
            return trianglesCreated;
        }

        auto vertices = getTriangleVertices(mesh, currentTri);
        if (vertices[0] == 0 && vertices[1] == 0 && vertices[2] == 0)
        {
            spdlog::debug("refineTriangle: triangle (edge={}, side={}) has invalid vertices", currentTri.edgeId,
                          currentTri.side);
            return trianglesCreated;
        }

        size_t longestEdgeId = findLongestEdge(mesh, currentTri);
        if (longestEdgeId == 0)
        {
            spdlog::debug("refineTriangle: could not find longest edge for triangle (edge={}, side={})",
                          currentTri.edgeId, currentTri.side);
            return trianglesCreated;
        }

        const auto *longestEdge = mesh.getEdge(longestEdgeId);
        if (!longestEdge)
            return trianglesCreated;

        // conforming refinement: bisect at the longest edge, refining the neighbor first if needed
        if (!longestEdge->payload.border)
        {
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
                TriangleId neighbor = {longestEdgeId, 1 - ourSide};

                size_t neighborLongest = findLongestEdge(mesh, neighbor);
                if (neighborLongest != 0 && neighborLongest != longestEdgeId)
                {
                    size_t createdByRecursion = refineTriangle(mesh, neighbor, maxDepth - 1);
                    trianglesCreated += createdByRecursion;

                    // No progress from recursion: retrying would spin on unchanged state.
                    if (createdByRecursion == 0)
                        return trianglesCreated;

                    // The recursive refinement may have bisected our longest edge,
                    // invalidating this triangle — re-locate via original vertices
                    Eigen::Vector3d center = Eigen::Vector3d::Zero();
                    for (int i = 0; i < 3; i++)
                    {
                        const auto *n = mesh.getNode(vertices[i]);
                        if (n)
                            center += n->payload.location;
                    }
                    center /= 3.0;

                    TriangleId newTri = findTriangleNearVertices(mesh, vertices, center.x(), center.y());
                    if (newTri.edgeId == 0)
                        return trianglesCreated;

                    // Retry with the re-located triangle without consuming depth
                    currentTri = newTri;
                    continue;
                }
            }
        }

        bool wasBorder = longestEdge->payload.border;
        auto result = bisectEdge(mesh, longestEdgeId);
        if (result.newVertexId != 0)
        {
            // We created 2 new triangles on each side (4 total, or 2 if border)
            trianglesCreated += wasBorder ? 2 : 4;
        }

        return trianglesCreated;
    }
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
            return {0, 0};

        const auto *n0 = _mesh.getNode(verts[0]);
        const auto *n1 = _mesh.getNode(verts[1]);
        const auto *n2 = _mesh.getNode(verts[2]);
        if (!n0 || !n1 || !n2)
            return {0, 0};

        const auto &p0 = n0->payload.location;
        const auto &p1 = n1->payload.location;
        const auto &p2 = n2->payload.location;

        auto sign = [](double px, double py, double ax, double ay, double bx, double by) {
            return (px - bx) * (ay - by) - (ax - bx) * (py - by);
        };

        double d0 = sign(x, y, p0.x(), p0.y(), p1.x(), p1.y()); // edge v0-v1
        double d1 = sign(x, y, p1.x(), p1.y(), p2.x(), p2.y()); // edge v1-v2
        double d2 = sign(x, y, p2.x(), p2.y(), p0.x(), p0.y()); // edge v2-v0

        bool hasNeg = (d0 < 0) || (d1 < 0) || (d2 < 0);
        bool hasPos = (d0 > 0) || (d1 > 0) || (d2 > 0);

        if (!(hasNeg && hasPos))
            return current;

        // Determine expected sign (majority vote) and pick the most-violated edge
        int negCount = (d0 < 0) + (d1 < 0) + (d2 < 0);
        bool expectPositive = negCount < 2;

        double worstVal = 0;
        int worstEdge = -1;

        auto checkEdge = [&](int edgeIdx, double d) {
            if (d == 0)
            {
                worstVal = 0.000001;
                worstEdge = edgeIdx;
                return;
            }
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
            return {0, 0};

        // Cross the worst edge to the neighboring triangle
        // Edge 0: verts[0]-verts[1] = the defining edge of current TriangleId
        // Edge 1: verts[1]-verts[2]
        // Edge 2: verts[2]-verts[0]
        TriangleId neighbor{0, 0};

        if (worstEdge == 0)
        {
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
            return {0, 0}; // Hit border or invalid edge

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

    struct TrianglePlane
    {
        Eigen::Vector3d normal;
        Eigen::Vector3d origin;
    };

    spdlog::debug("countPointsPerTriangle: mesh has {} nodes, {} edges", mesh.size_nodes(), mesh.size_edges());

    TriangleLocator locator(mesh);

    ankerl::unordered_dense::map<TriangleId, TrianglePlane, TriangleIdHash> planeCache;
    for (auto it = mesh.cedgebegin(); it != mesh.cedgeend(); ++it)
    {
        for (int side = 0; side < 2; side++)
        {
            if (side == 1 && it->second.payload.border)
                continue;

            TriangleId tri{it->first, side};
            auto verts = getTriangleVertices(mesh, tri);
            if (verts[0] == 0 && verts[1] == 0 && verts[2] == 0)
                continue;

            const auto *n0 = mesh.getNode(verts[0]);
            const auto *n1 = mesh.getNode(verts[1]);
            const auto *n2 = mesh.getNode(verts[2]);
            if (n0 && n1 && n2)
            {
                Eigen::Vector3d normal = (n1->payload.location - n0->payload.location)
                                             .cross(n2->payload.location - n0->payload.location)
                                             .normalized();
                planeCache[tri] = {normal, n0->payload.location};
            }
        }
    }

    std::vector<const Eigen::Vector3d *> allPoints;
    {
        size_t totalPts = 0;
        for (const auto &cloud : points)
            totalPts += cloud.size();
        allPoints.reserve(totalPts);
    }
    for (const auto &cloud : points)
        for (const auto &p : cloud)
            allPoints.push_back(&p);

    ankerl::unordered_dense::map<TriangleId, Accumulator, TriangleIdHash> accumulators;

    const int num_points = static_cast<int>(allPoints.size());
#pragma omp parallel
    {
        ankerl::unordered_dense::map<TriangleId, Accumulator, TriangleIdHash> localAcc;

#pragma omp for schedule(dynamic, 256) // NOLINT(modernize-loop-convert)
        for (int pi = 0; pi < num_points; pi++)
        {
            const auto &p = *allPoints[pi];
            TriangleId tri = locator.find(p.x(), p.y());
            if (tri.edgeId == 0)
                continue;
            tri = lowestEdgeTriangleId(mesh, tri);

            auto &acc = localAcc[tri];
            acc.count++;

            auto planeIt = planeCache.find(tri);
            if (planeIt != planeCache.end())
            {
                double dist = (p - planeIt->second.origin).dot(planeIt->second.normal);
                acc.sumDist += dist;
                acc.sumDistSq += dist * dist;
            }
        }

#pragma omp critical(merge_point_accumulators)
        {
            for (const auto &[tri, lacc] : localAcc)
            {
                auto &acc = accumulators[tri];
                acc.count += lacc.count;
                acc.sumDist += lacc.sumDist;
                acc.sumDistSq += lacc.sumDistSq;
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
                            double minDistanceVariance, int maxIterations, double minTriangleSizeMeters)
{
    size_t totalCreated = 0;

    for (int iter = 0; iter < maxIterations; iter++)
    {
        auto stats = countPointsPerTriangle(mesh, points);

        std::vector<std::pair<TriangleId, std::array<size_t, 3>>> toRefine;
        size_t skippedSmall = 0;
        for (const auto &[tri, s] : stats)
        {
            if (s.count > maxPointsPerTriangle && s.distanceVariance > minDistanceVariance)
            {
                const auto verts = getTriangleVertices(mesh, tri);
                if (minTriangleSizeMeters > 0.0)
                {
                    const auto *n0 = mesh.getNode(verts[0]);
                    const auto *n1 = mesh.getNode(verts[1]);
                    const auto *n2 = mesh.getNode(verts[2]);
                    if (n0 && n1 && n2)
                    {
                        Eigen::Vector2d p0 = n0->payload.location.head<2>();
                        Eigen::Vector2d p1 = n1->payload.location.head<2>();
                        Eigen::Vector2d p2 = n2->payload.location.head<2>();
                        double maxEdge = std::max({(p0 - p1).norm(), (p1 - p2).norm(), (p2 - p0).norm()});
                        if (maxEdge < minTriangleSizeMeters)
                        {
                            skippedSmall++;
                            spdlog::debug("refineByPointDensity: skipping triangle at min size "
                                          "(edge={}, side={}), {:.4f}m < {:.4f}m limit",
                                          tri.edgeId, tri.side, maxEdge, minTriangleSizeMeters);
                            continue;
                        }
                    }
                }
                toRefine.emplace_back(tri, verts);
                spdlog::debug("refineByPointDensity: triangle (edge={}, side={}) has {} points, variance {}",
                              tri.edgeId, tri.side, s.count, s.distanceVariance);
            }
        }

        if (toRefine.empty())
        {
            spdlog::info("refineByPointDensity: converged after {} iterations, {} triangles created, "
                         "{} at minimum size limit",
                         iter, totalCreated, skippedSmall);
            break;
        }

        spdlog::info("refineByPointDensity: iteration {}, refining {} triangles exceeding {} points", iter,
                     toRefine.size(), maxPointsPerTriangle);

        size_t createdThisIter = 0;
        for (const auto &[tri, queuedVerts] : toRefine)
        {
            if (getTriangleVertices(mesh, tri) != queuedVerts)
            {
                spdlog::debug("refineByPointDensity: skipping triangle already split (edge={}, side={})", tri.edgeId,
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

MeshTriangle sortedTriangle(size_t a, size_t b, size_t c)
{
    MeshTriangle t{a, b, c};
    std::sort(t.begin(), t.end());
    return t;
}

namespace
{
Eigen::Vector3d barycentricInXy(const Eigen::Vector2d &p, const Eigen::Vector2d &a, const Eigen::Vector2d &b,
                                const Eigen::Vector2d &c)
{
    const Eigen::Vector2d v0 = b - a, v1 = c - a, v2 = p - a;
    const double d = v0.x() * v1.y() - v1.x() * v0.y();
    if (std::abs(d) < 1e-12)
        return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    const double l1 = (v2.x() * v1.y() - v1.x() * v2.y()) / d;
    const double l2 = (v0.x() * v2.y() - v2.x() * v0.y()) / d;
    return Eigen::Vector3d(1 - l1 - l2, l1, l2);
}
} // namespace

std::vector<MeshPointSample> sampleMeshPoints(const MeshGraph &mesh, const std::vector<point_cloud> &points)
{
    if (mesh.size_nodes() == 0)
        return {};
    std::vector<const Eigen::Vector3d *> allPoints;
    for (const auto &cloud : points)
        for (const auto &p : cloud)
            allPoints.push_back(&p);

    std::vector<MeshPointSample> samples(allPoints.size());
    std::vector<char> valid(allPoints.size(), 0);
    TriangleLocator locator(mesh);
    const int num_points = static_cast<int>(allPoints.size());
#pragma omp parallel for schedule(dynamic, 256) // NOLINT(modernize-loop-convert)
    for (int pi = 0; pi < num_points; pi++)
    {
        const auto &p = *allPoints[pi];
        const TriangleId tri = locator.find(p.x(), p.y());
        if (tri.edgeId == 0)
            continue;
        const auto verts = getTriangleVertices(mesh, tri);
        const auto *n0 = mesh.getNode(verts[0]);
        const auto *n1 = mesh.getNode(verts[1]);
        const auto *n2 = mesh.getNode(verts[2]);
        if (!n0 || !n1 || !n2)
            continue;
        const Eigen::Vector3d bary = barycentricInXy(p.head<2>(), n0->payload.location.head<2>(),
                                                     n1->payload.location.head<2>(), n2->payload.location.head<2>());
        if (!bary.allFinite())
            continue;
        samples[pi] = MeshPointSample{verts, bary, p.z()};
        valid[pi] = 1;
    }

    size_t dst = 0;
    for (size_t i = 0; i < samples.size(); i++)
        if (valid[i])
            samples[dst++] = samples[i];
    samples.resize(dst);
    return samples;
}

std::vector<MeshBend> meshBendsBetweenDataTriangles(const MeshGraph &mesh, const MeshTriangleSet &dataTriangles)
{
    std::vector<MeshBend> bends;
    for (auto it = mesh.cedgebegin(); it != mesh.cedgeend(); ++it)
    {
        const MeshEdge &edge = it->second.payload;
        if (edge.border)
            continue;
        const std::array<size_t, 4> ids{it->second.getSource(), it->second.getDest(), edge.triangleOppositeNodes[0],
                                        edge.triangleOppositeNodes[1]};
        if (!dataTriangles.contains(sortedTriangle(ids[0], ids[1], ids[2])) ||
            !dataTriangles.contains(sortedTriangle(ids[0], ids[1], ids[3])))
            continue;
        std::array<Eigen::Vector2d, 4> xy;
        bool ok = true;
        for (int k = 0; k < 4 && ok; k++)
        {
            const auto *node = mesh.getNode(ids[k]);
            ok = node != nullptr;
            if (ok)
                xy[k] = node->payload.location.head<2>();
        }
        const Eigen::Vector2d ab = xy[1] - xy[0];
        if (!ok || ab.norm() == 0)
            continue;
        auto distToAB = [&](const Eigen::Vector2d &p) {
            const Eigen::Vector2d ap = p - xy[0];
            return std::abs(ab.x() * ap.y() - ab.y() * ap.x()) / ab.norm();
        };
        const double meanOppositeDistance = 0.5 * (distToAB(xy[2]) + distToAB(xy[3]));
        if (meanOppositeDistance > 0)
            bends.push_back(MeshBend{ids, meanOppositeDistance});
    }
    return bends;
}

double estimatePointHeightSigma(const std::vector<point_cloud> &points, size_t neighbours, size_t maxSampledPoints)
{
    jk::tree::KDTree<double, 2> tree;
    size_t total = 0;
    for (const auto &cloud : points)
        for (const auto &p : cloud)
        {
            tree.addPoint({p.x(), p.y()}, p.z(), false);
            total++;
        }
    if (total <= neighbours)
        return 0;
    tree.splitOutstanding();

    const size_t stride = std::max<size_t>(1, total / maxSampledPoints);
    std::vector<double> deviations, heights;
    size_t i = 0;
    for (const auto &cloud : points)
        for (const auto &p : cloud)
        {
            if (i++ % stride != 0)
                continue;
            heights.clear();
            const size_t neighboursIncludingSelf = neighbours + 1;
            for (const auto &n : tree.searchKnn({p.x(), p.y()}, neighboursIncludingSelf))
                heights.push_back(n.payload);
            auto mid = heights.begin() + heights.size() / 2;
            std::nth_element(heights.begin(), mid, heights.end());
            deviations.push_back(std::abs(p.z() - *mid));
        }
    auto mid = deviations.begin() + deviations.size() / 2;
    std::nth_element(deviations.begin(), mid, deviations.end());
    constexpr double MAD_TO_GAUSSIAN_SIGMA = 1.4826;
    return MAD_TO_GAUSSIAN_SIGMA * *mid;
}

std::vector<point_cloud> filterPointsWithoutHeightAgreement(const std::vector<point_cloud> &points,
                                                            double heightTolerance, size_t neighbours,
                                                            size_t minAgreeingNeighbours)
{
    jk::tree::KDTree<double, 2> tree;
    for (const auto &cloud : points)
        for (const auto &p : cloud)
            tree.addPoint({p.x(), p.y()}, p.z(), false);
    tree.splitOutstanding();

    const size_t neighboursIncludingSelf = neighbours + 1;
    const size_t minAgreeingIncludingSelf = minAgreeingNeighbours + 1;
    std::vector<point_cloud> agreeing(points.size());
    for (size_t c = 0; c < points.size(); c++)
        for (const auto &p : points[c])
        {
            size_t agreeingIncludingSelf = 0;
            for (const auto &n : tree.searchKnn({p.x(), p.y()}, neighboursIncludingSelf))
                agreeingIncludingSelf += std::abs(n.payload - p.z()) < heightTolerance;
            if (agreeingIncludingSelf >= minAgreeingIncludingSelf)
                agreeing[c].push_back(p);
        }
    return agreeing;
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

    surface_model result;
    result.mesh = surfaces[0].mesh;

    ankerl::unordered_dense::map<size_t, std::pair<Eigen::Vector3d, double>> vertexWeights;

    for (auto it = result.mesh.cnodebegin(); it != result.mesh.cnodeend(); ++it)
    {
        vertexWeights[it->first] = {Eigen::Vector3d::Zero(), 0.0};
    }

    std::vector<ankerl::unordered_dense::map<size_t, std::pair<Eigen::Vector3d, double>>> threadLocalWeights(
        surfaces.size());

#pragma omp parallel for schedule(dynamic)
    for (size_t surfIdx = 0; surfIdx < surfaces.size(); surfIdx++)
    {
        const auto &surf = surfaces[surfIdx];

        if (surf.mesh.size_nodes() == 0)
        {
            continue;
        }

        auto triangleCounts = countPointsPerTriangle(surf.mesh, surf.cloud);

        ankerl::unordered_dense::map<size_t, size_t> vertexPointCounts;

        for (const auto &[tri, triStats] : triangleCounts)
        {
            auto verts = getTriangleVertices(surf.mesh, tri);
            if (verts[0] == 0 && verts[1] == 0 && verts[2] == 0)
            {
                continue;
            }

            for (int i = 0; i < 3; i++)
            {
                vertexPointCounts[verts[i]] += triStats.count;
            }
        }

        auto &threadLocal = threadLocalWeights[surfIdx];
        for (const auto &[nodeId, pointCount] : vertexPointCounts)
        {
            const auto *nodePtr = surf.mesh.getNode(nodeId);
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
