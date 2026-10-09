#include <opencalibration/surface/intersect.hpp>

#include <opencalibration/geometry/intersection.hpp>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <iterator>

namespace opencalibration
{
bool MeshIntersectionSearcher::init(const MeshGraph &meshGraph, const IntersectionInfo &info)
{
    if (_meshGraph != &meshGraph)
    {
        _lastIntersection = {};
    }
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
    return init(*_meshGraph, _lastIntersection);
}

bool MeshIntersectionSearcher::reinit(const IntersectionInfo &info)
{
    return _meshGraph != nullptr && init(*_meshGraph, info);
}

namespace
{
using Info = MeshIntersectionSearcher::IntersectionInfo;

double orient2d(const Eigen::Vector2d &p, const Eigen::Vector2d &q, const Eigen::Vector2d &x)
{
    return (q.x() - p.x()) * (x.y() - p.y()) - (q.y() - p.y()) * (x.x() - p.x());
}

Eigen::Vector2d cornerXY(const Info &info, int i)
{
    return info.nodeLocations[i % 3]->head<2>();
}

void makeAnticlockwise(Info &info)
{
    if (orient2d(cornerXY(info, 0), cornerXY(info, 1), cornerXY(info, 2)) < 0)
    {
        std::swap(info.nodeIndexes[0], info.nodeIndexes[1]);
        std::swap(info.nodeLocations[0], info.nodeLocations[1]);
    }
}

bool crossEdge(const MeshGraph &meshGraph, Info &info, int edgeIndex, size_t maxSteps)
{
    const size_t a = info.nodeIndexes[edgeIndex], b = info.nodeIndexes[(edgeIndex + 1) % 3];
    const MeshGraph::Edge *edge = meshGraph.getEdge(a, b);
    if (edge == nullptr)
        edge = meshGraph.getEdge(b, a);
    if (edge == nullptr)
    {
        info.type = Info::GRAPH_STRUCTURE_INCONSISTENT;
        return false;
    }
    if (edge->payload.border)
    {
        info.type = Info::OUTSIDE_BORDER;
        return false;
    }

    const int replaced = (edgeIndex + 2) % 3;
    const auto &opposite = edge->payload.triangleOppositeNodes;
    if (opposite[0] == info.nodeIndexes[replaced])
        info.nodeIndexes[replaced] = opposite[1];
    else if (opposite[1] == info.nodeIndexes[replaced])
        info.nodeIndexes[replaced] = opposite[0];
    else
    {
        info.type = Info::GRAPH_STRUCTURE_INCONSISTENT;
        return false;
    }
    info.nodeLocations[replaced] = &meshGraph.getNode(info.nodeIndexes[replaced])->payload.location;

    if (++info.steps > maxSteps)
    {
        info.type = Info::MAX_STEPS_EXCEEDED;
        return false;
    }
    return true;
}
int exitEdgeTowards(const Info &info, const Eigen::Vector2d &from, const Eigen::Vector2d &to)
{
    const auto isOutside = [&](int i) { return orient2d(cornerXY(info, i), cornerXY(info, i + 1), to) < 0; };
    for (int i = 0; i < 3; i++)
        if (isOutside(i) && orient2d(from, to, cornerXY(info, i)) * orient2d(from, to, cornerXY(info, i + 1)) <= 0)
            return i;
    for (int i = 0; i < 3; i++)
        if (isOutside(i))
            return i;
    return -1;
}

struct EdgeSpan
{
    double enter = -INFINITY, exit = INFINITY;
    int enterEdge = -1, exitEdge = -1;
};

EdgeSpan spanAlong(const Info &info, const Eigen::Vector2d &origin, const Eigen::Vector2d &dir)
{
    EdgeSpan span;
    for (int i = 0; i < 3; i++)
    {
        const Eigen::Vector2d p = cornerXY(info, i), q = cornerXY(info, i + 1);
        const double rate = (q - p).x() * dir.y() - (q - p).y() * dir.x();
        const double t = -orient2d(p, q, origin) / rate;
        if (rate > 0 && t > span.enter)
        {
            span.enter = t;
            span.enterEdge = i;
        }
        if (rate < 0 && t < span.exit)
        {
            span.exit = t;
            span.exitEdge = i;
        }
    }
    return span;
}

struct TrianglePlane
{
    explicit TrianglePlane(const Info &info)
        : corner(*info.nodeLocations[0]),
          normal((*info.nodeLocations[1] - corner).cross(*info.nodeLocations[2] - corner))
    {
    }

    double heightAbove(const Eigen::Vector3d &x) const
    {
        return normal.dot(x - corner) / normal.z();
    }

    Eigen::Vector3d corner, normal;
};

bool walkToRayLine(const MeshGraph &meshGraph, Info &info, const Eigen::Vector2d &from, const Eigen::Vector2d &to,
                   size_t maxSteps)
{
    while (true)
    {
        makeAnticlockwise(info);
        const int exitEdge = exitEdgeTowards(info, from, to);
        if (exitEdge < 0)
            return true;
        if (!crossEdge(meshGraph, info, exitEdge, maxSteps))
            return false;
    }
}

bool walkDownhillAlongRay(const MeshGraph &meshGraph, Info &info, const Eigen::Vector3d &onRay,
                          const Eigen::Vector3d &downhill, size_t maxSteps)
{
    while (true)
    {
        makeAnticlockwise(info);
        const EdgeSpan span = spanAlong(info, onRay.head<2>(), downhill.head<2>());
        if (span.enterEdge < 0 || span.exitEdge < 0)
            return true;

        const TrianglePlane plane(info);
        const double heightIn = plane.heightAbove(onRay + span.enter * downhill);
        const double heightOut = plane.heightAbove(onRay + span.exit * downhill);
        const int nextEdge = heightIn > 0 && heightOut > 0   ? span.exitEdge
                             : heightIn < 0 && heightOut < 0 ? span.enterEdge
                                                             : -1;
        if (nextEdge < 0)
            return true;
        if (!crossEdge(meshGraph, info, nextEdge, maxSteps))
            return false;
    }
}

bool intersectTrianglePlane(const ray_d &r, Info &info)
{
    plane_3_corners_d plane;
    for (size_t i = 0; i < 3; i++)
        plane.corner[i] = *info.nodeLocations[i];
    return rayPlaneIntersection(r, cornerPlane2normOffsetPlane(plane), info.intersectionLocation) &&
           !info.intersectionLocation.hasNaN();
}

bool moveToNearestCrossing(const MeshGraph &meshGraph, Info &info, const ray_d &r, size_t maxSteps)
{
    Info walker = info;
    bool moved = false, foundNearer = false;
    while (true)
    {
        makeAnticlockwise(walker);
        const EdgeSpan span = spanAlong(walker, r.offset.head<2>(), r.dir.head<2>());
        if (moved)
        {
            const TrianglePlane plane(walker);
            const auto heightAt = [&](double t) { return plane.heightAbove(r.offset + t * r.dir); };
            if (heightAt(std::max(span.enter, 0.)) * heightAt(span.exit) <= 0)
            {
                info.nodeIndexes = walker.nodeIndexes;
                info.nodeLocations = walker.nodeLocations;
                foundNearer = true;
            }
        }
        if (span.enterEdge < 0 || span.enter <= 0 || !crossEdge(meshGraph, walker, span.enterEdge, maxSteps))
            break;
        moved = true;
    }
    info.steps = walker.steps;
    return !foundNearer || intersectTrianglePlane(r, info);
}
} // namespace

const MeshIntersectionSearcher::IntersectionInfo &MeshIntersectionSearcher::triangleIntersect(const ray_d &r)
{
    if (_meshGraph == nullptr)
    {
        _info.type = IntersectionInfo::GRAPH_STRUCTURE_INCONSISTENT;
        return _info;
    }

    const bool previous_walk_failed_midway = _info.type == IntersectionInfo::MAX_STEPS_EXCEEDED;
    if (previous_walk_failed_midway && _lastIntersection.type == IntersectionInfo::INTERSECTION)
        _info = _lastIntersection;

    _info.type = IntersectionInfo::PENDING;
    _info.steps = 0;
    _info.intersectionLocation.fill(NAN);
    const size_t maxWalkSteps = 100 + 2 * _meshGraph->size_nodes();

    const Eigen::Vector3d centroid = (*_info.nodeLocations[0] + *_info.nodeLocations[1] + *_info.nodeLocations[2]) / 3;
    const double centroidT = r.dir.z() != 0 ? (centroid.z() - r.offset.z()) / r.dir.z() : 0;
    const Eigen::Vector3d onRay = r.offset + centroidT * r.dir;
    const double horizontal = r.dir.head<2>().norm();
    const Eigen::Vector3d downhill = (r.dir.z() > 0 ? -r.dir : r.dir) / horizontal;

    if (!walkToRayLine(*_meshGraph, _info, centroid.head<2>(), onRay.head<2>(), maxWalkSteps))
        return _info;
    if (horizontal > 0 && !walkDownhillAlongRay(*_meshGraph, _info, onRay, downhill, maxWalkSteps))
        return _info;

    if (!intersectTrianglePlane(r, _info))
    {
        _info.type = IntersectionInfo::RAY_PARALLEL_TO_PLANE;
        return _info;
    }

    const bool crossingMayBeBehindNearest = horizontal > 0 && (_info.intersectionLocation - r.offset).dot(r.dir) > 0;
    if (crossingMayBeBehindNearest && !moveToNearestCrossing(*_meshGraph, _info, r, maxWalkSteps))
    {
        _info.type = IntersectionInfo::RAY_PARALLEL_TO_PLANE;
        return _info;
    }
    _info.type = IntersectionInfo::INTERSECTION;
    _lastIntersection = _info;
    return _info;
}

const MeshIntersectionSearcher::IntersectionInfo &MeshIntersectionSearcher::lastResult()
{
    return _info;
}

struct MeshLineOfSight::MaxMipmap
{
    struct Level
    {
        Level(int cols, int rows, double cellSize)
            : cols(cols), rows(rows), cellSize(cellSize),
              maxHeight(static_cast<size_t>(cols) * rows, -std::numeric_limits<double>::infinity()),
              maxGradient(maxHeight.size(), 0)
        {
        }

        int cols, rows;
        double cellSize;
        std::vector<double> maxHeight, maxGradient;

        [[nodiscard]] size_t index(int col, int row) const
        {
            return static_cast<size_t>(row) * cols + col;
        }

        [[nodiscard]] bool contains(const Eigen::Vector2d &cell) const
        {
            return cell.x() >= 0 && cell.y() >= 0 && cell.x() < cols && cell.y() < rows;
        }

        void include(size_t i, double height, double gradient)
        {
            maxHeight[i] = std::max(maxHeight[i], height);
            maxGradient[i] = std::max(maxGradient[i], gradient);
        }

        [[nodiscard]] Level downsampled() const
        {
            Level coarse((cols + 1) / 2, (rows + 1) / 2, cellSize * 2);
            for (int row = 0; row < rows; row++)
                for (int col = 0; col < cols; col++)
                    coarse.include(coarse.index(col / 2, row / 2), maxHeight[index(col, row)],
                                   maxGradient[index(col, row)]);
            return coarse;
        }
    };

    Eigen::Vector2d origin;
    std::vector<Level> levels;
};

struct MeshLineOfSight::ShadowRay
{
    Eigen::Vector3d origin;
    Eigen::Vector2d directionXY;
    double gradient;

    [[nodiscard]] Eigen::Vector2d xy(double t) const
    {
        return origin.head<2>() + t * directionXY;
    }

    [[nodiscard]] double height(double t) const
    {
        return origin.z() + t * gradient;
    }
};

namespace
{
struct Triangle
{
    Eigen::Vector3d a, b, c;
};

bool hasEdge(const MeshGraph &meshGraph, size_t a, size_t b)
{
    return meshGraph.getEdge(a, b) != nullptr || meshGraph.getEdge(b, a) != nullptr;
}

template <typename Visit> void forEachTriangle(const MeshGraph &meshGraph, Visit &&visit)
{
    for (auto it = meshGraph.cedgebegin(); it != meshGraph.cedgeend(); ++it)
    {
        const size_t source = it->second.getSource(), dest = it->second.getDest();
        for (size_t opposite : it->second.payload.triangleOppositeNodes)
        {
            const auto *oppositeNode = meshGraph.getNode(opposite);
            if (oppositeNode == nullptr || !hasEdge(meshGraph, source, opposite) || !hasEdge(meshGraph, dest, opposite))
                continue;
            visit(Triangle{meshGraph.getNode(source)->payload.location, meshGraph.getNode(dest)->payload.location,
                           oppositeNode->payload.location});
        }
    }
}

double medianHorizontalEdgeLength(const MeshGraph &meshGraph)
{
    std::vector<double> lengths;
    lengths.reserve(meshGraph.size_edges());
    for (auto it = meshGraph.cedgebegin(); it != meshGraph.cedgeend(); ++it)
    {
        const Eigen::Vector3d &source = meshGraph.getNode(it->second.getSource())->payload.location;
        const Eigen::Vector3d &dest = meshGraph.getNode(it->second.getDest())->payload.location;
        lengths.push_back((dest - source).head<2>().norm());
    }
    if (lengths.empty())
        return NAN;
    auto median = lengths.begin() + lengths.size() / 2;
    std::nth_element(lengths.begin(), median, lengths.end());
    return *median;
}

Eigen::Vector2d planeGradient(const Triangle &triangle)
{
    const Eigen::Vector3d normal = (triangle.b - triangle.a).cross(triangle.c - triangle.a);
    if (std::abs(normal.z()) <= 1e-12 * normal.norm())
        return Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
    return -normal.head<2>() / normal.z();
}

double maxHeightOver(const Triangle &triangle, const Eigen::Vector2d &gradient, const Eigen::AlignedBox2d &region)
{
    const double vertexMaxHeight = std::max({triangle.a.z(), triangle.b.z(), triangle.c.z()});
    const Eigen::Vector2d supportCorner(gradient.x() > 0 ? region.max().x() : region.min().x(),
                                        gradient.y() > 0 ? region.max().y() : region.min().y());
    const double planeMaxHeight = triangle.a.z() + gradient.dot(supportCorner - triangle.a.head<2>());
    if (!std::isfinite(planeMaxHeight))
        return vertexMaxHeight;
    return std::min(vertexMaxHeight, planeMaxHeight + 1e-9 * std::abs(planeMaxHeight));
}

Eigen::Vector2d cellAhead(const Eigen::Vector2d &p, const Eigen::Vector2d &direction, double cellSize)
{
    const Eigen::Vector2d scaled = p / cellSize;
    return {direction.x() < 0 ? std::ceil(scaled.x()) - 1 : std::floor(scaled.x()),
            direction.y() < 0 ? std::ceil(scaled.y()) - 1 : std::floor(scaled.y())};
}

double distanceToCellExit(const Eigen::Vector2d &p, const Eigen::Vector2d &direction, const Eigen::Vector2d &cell,
                          double cellSize)
{
    double tExit = std::numeric_limits<double>::infinity();
    for (int axis = 0; axis < 2; axis++)
        if (direction[axis] != 0)
        {
            const double exitPlane = (cell[axis] + (direction[axis] > 0 ? 1 : 0)) * cellSize;
            tExit = std::min(tExit, (exitPlane - p[axis]) / direction[axis]);
        }
    return std::max(0.0, tExit);
}
} // namespace

bool MeshLineOfSight::init(const MeshGraph &meshGraph)
{
    _marchStep = NAN;
    _maxMipmap.reset();
    if (!_searcher.init(meshGraph))
        return false;

    Eigen::AlignedBox3d bounds;
    for (auto it = meshGraph.cnodebegin(); it != meshGraph.cnodeend(); ++it)
        bounds.extend(it->second.payload.location);
    _maxSurfaceZ = bounds.max().z();

    constexpr double MARCH_STEPS_PER_EDGE = 8;
    constexpr double EDGES_PER_CELL = 0.5;
    const double edgeLength = medianHorizontalEdgeLength(meshGraph);
    _marchStep = edgeLength / MARCH_STEPS_PER_EDGE;
    if (!(_marchStep > 0))
        return false;

    auto mipmap = std::make_shared<MaxMipmap>();
    mipmap->origin = bounds.min().head<2>();
    const double cellSize = edgeLength * EDGES_PER_CELL;
    const Eigen::Vector2d extent = bounds.max().head<2>() - mipmap->origin;
    MaxMipmap::Level base(static_cast<int>(extent.x() / cellSize) + 1, static_cast<int>(extent.y() / cellSize) + 1,
                          cellSize);

    forEachTriangle(meshGraph, [&](const Triangle &triangle) {
        const Eigen::Vector2d gradient = planeGradient(triangle);
        Eigen::AlignedBox2d footprint(triangle.a.head<2>());
        footprint.extend(triangle.b.head<2>()).extend(triangle.c.head<2>());
        const Eigen::Vector2d lo = (footprint.min() - mipmap->origin) / cellSize;
        const Eigen::Vector2d hi = (footprint.max() - mipmap->origin) / cellSize;
        for (int row = std::max(0, static_cast<int>(lo.y())); row <= std::min(base.rows - 1, static_cast<int>(hi.y()));
             row++)
            for (int col = std::max(0, static_cast<int>(lo.x()));
                 col <= std::min(base.cols - 1, static_cast<int>(hi.x())); col++)
            {
                const Eigen::Vector2d cellMin = mipmap->origin + Eigen::Vector2d(col, row) * cellSize;
                const Eigen::AlignedBox2d cellBox(cellMin, cellMin + Eigen::Vector2d::Constant(cellSize));
                base.include(base.index(col, row), maxHeightOver(triangle, gradient, cellBox.intersection(footprint)),
                             gradient.norm());
            }
    });

    mipmap->levels.push_back(std::move(base));
    while (mipmap->levels.back().cols > 1 || mipmap->levels.back().rows > 1)
        mipmap->levels.push_back(mipmap->levels.back().downsampled());
    _maxMipmap = std::move(mipmap);
    return true;
}

bool MeshLineOfSight::traceUnoccluded(const ShadowRay &ray, double tMax, double epsilon)
{
    const auto &levels = _maxMipmap->levels;
    const double tNudge = 1e-6 * levels.front().cellSize;

    double t = 0;
    while (t < tMax)
    {
        const Eigen::Vector2d p = ray.xy(t) - _maxMipmap->origin;
        for (auto level = levels.rbegin(); level != levels.rend(); ++level)
        {
            const Eigen::Vector2d cell = cellAhead(p, ray.directionXY, level->cellSize);
            if (!level->contains(cell))
                return true;
            const size_t i = level->index(static_cast<int>(cell.x()), static_cast<int>(cell.y()));
            const bool rayAboveCell = level->maxHeight[i] <= ray.height(t) + epsilon;
            const bool rayOutclimbsCell = level->maxGradient[i] <= ray.gradient;
            const bool cellMayOcclude = !rayAboveCell && !rayOutclimbsCell;
            const bool finestLevel = std::next(level) == levels.rend();
            if (cellMayOcclude && !finestLevel)
                continue;

            const double tExit = t + distanceToCellExit(p, ray.directionXY, cell, level->cellSize);
            if (cellMayOcclude && !marchUnoccluded(ray, t, tExit, epsilon))
                return false;
            t = tExit + tNudge;
            break;
        }
    }
    return true;
}

bool MeshLineOfSight::marchUnoccluded(const ShadowRay &ray, double tStart, double tEnd, double epsilon)
{
    const int steps = std::max(1, static_cast<int>(std::ceil((tEnd - tStart) / _marchStep)));
    for (int step = 1; step <= steps; step++)
    {
        const double t = tStart + (tEnd - tStart) * step / steps;
        if (surfaceHeight(ray.xy(t)) > ray.height(t) + epsilon)
            return false;
    }
    return true;
}

double MeshLineOfSight::surfaceHeight(const Eigen::Vector2d &xy)
{
    if (_searcher.lastResult().type != MeshIntersectionSearcher::IntersectionInfo::INTERSECTION && !_searcher.reinit())
        return NAN;

    const ray_d down{{0, 0, -1}, {xy.x(), xy.y(), _maxSurfaceZ + 1}};
    const auto &hit = _searcher.triangleIntersect(down);
    return hit.type == MeshIntersectionSearcher::IntersectionInfo::INTERSECTION ? hit.intersectionLocation.z() : NAN;
}

bool MeshLineOfSight::surfaceVisibleFrom(const Eigen::Vector2d &xy, const Eigen::Vector3d &viewpoint)
{
    if (!(_marchStep > 0))
        return true;

    const double z = surfaceHeight(xy);
    if (std::isnan(z))
        return true;

    return visibleFromSurfacePoint({xy.x(), xy.y(), z}, _searcher.lastResult(), viewpoint);
}

bool MeshLineOfSight::surfaceVisibleFrom(const MeshIntersectionSearcher::IntersectionInfo &surfaceHit,
                                         const Eigen::Vector3d &viewpoint)
{
    if (!(_marchStep > 0))
        return true;
    return visibleFromSurfacePoint(surfaceHit.intersectionLocation, surfaceHit, viewpoint);
}

bool MeshLineOfSight::visibleFromSurfacePoint(const Eigen::Vector3d &origin,
                                              const MeshIntersectionSearcher::IntersectionInfo &originTriangle,
                                              const Eigen::Vector3d &viewpoint)
{
    const Eigen::Vector3d toViewpoint = viewpoint - origin;
    const double horizontalDistance = toViewpoint.head<2>().norm();
    if (toViewpoint.z() <= 0 || horizontalDistance == 0)
        return true;

    const ShadowRay ray{origin, toViewpoint.head<2>() / horizontalDistance, toViewpoint.z() / horizontalDistance};
    const double tMax = std::min(horizontalDistance, (_maxSurfaceZ - origin.z()) / ray.gradient);
    const MeshIntersectionSearcher::IntersectionInfo start = originTriangle;
    const bool visible = traceUnoccluded(ray, tMax, 1e-6 * toViewpoint.norm());
    static_cast<void>(_searcher.reinit(start));
    return visible;
}

std::vector<MeshLineOfSight> sightlinesOver(const std::vector<surface_model> &surfaces)
{
    std::vector<MeshLineOfSight> sightlines(surfaces.size());
    for (size_t si = 0; si < surfaces.size(); si++)
        if (!sightlines[si].init(surfaces[si].mesh))
            sightlines[si] = MeshLineOfSight();
    return sightlines;
}

bool surfaceVisibleFrom(std::vector<MeshLineOfSight> &sightlines, const Eigen::Vector3d &point,
                        const Eigen::Vector3d &viewpoint)
{
    return std::all_of(sightlines.begin(), sightlines.end(),
                       [&](auto &sightline) { return sightline.surfaceVisibleFrom(point.head<2>(), viewpoint); });
}

bool surfaceVisibleFrom(std::vector<MeshLineOfSight> &sightlines, size_t hitSurface,
                        const MeshIntersectionSearcher::IntersectionInfo &surfaceHit, const Eigen::Vector3d &viewpoint)
{
    for (size_t si = 0; si < sightlines.size(); si++)
        if (!(si == hitSurface ? sightlines[si].surfaceVisibleFrom(surfaceHit, viewpoint)
                               : sightlines[si].surfaceVisibleFrom(surfaceHit.intersectionLocation.head<2>(), viewpoint)))
            return false;
    return true;
}
} // namespace opencalibration
