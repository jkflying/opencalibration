#include <opencalibration/surface/expand_mesh.hpp>

#include <jk/KDTree.h>
#include <spdlog/spdlog.h>

#include <algorithm>

namespace
{
std::array<double, 2> toArray(const Eigen::Vector2d &vec)
{
    return {vec.x(), vec.y()};
}

struct MeshFootprint
{
    jk::tree::KDTree<double, 2> vertexTree, cameraTree;
    Eigen::Vector2d cameraMin, cameraMax;
    double medianCameraSpacing;
    double medianHeight;
    double borderWidth;
};

MeshFootprint computeFootprint(const opencalibration::point_cloud &cameraLocations,
                               const std::vector<opencalibration::surface_model> &previousSurfaces)
{
    constexpr double BORDER_PER_MEDIAN_HEIGHT = 2;

    MeshFootprint f;
    f.cameraMin = Eigen::Vector2d::Constant(std::numeric_limits<double>::max());
    f.cameraMax = -f.cameraMin;

    for (const auto &surface : previousSurfaces)
    {
        for (auto nodeIter = surface.mesh.cnodebegin(); nodeIter != surface.mesh.cnodeend(); ++nodeIter)
        {
            const Eigen::Vector3d p = nodeIter->second.payload.location;
            f.vertexTree.addPoint(toArray(p.topRows<2>()), p.z(), false);
        }
        for (const auto &cloud : surface.cloud)
        {
            for (const auto &p : cloud)
            {
                f.vertexTree.addPoint(toArray(p.topRows<2>()), p.z(), false);
            }
        }
    }
    f.vertexTree.splitOutstanding();

    std::vector<double> heights;
    heights.reserve(cameraLocations.size());

    for (const auto &p : cameraLocations)
    {
        f.cameraMin = f.cameraMin.cwiseMin(p.topRows<2>());
        f.cameraMax = f.cameraMax.cwiseMax(p.topRows<2>());
        f.cameraTree.addPoint(toArray(p.topRows<2>()), p.z(), false);

        if (f.vertexTree.size() > 0)
        {
            auto nearest = f.vertexTree.search(toArray(p.topRows<2>()));
            double agl = p.z() - nearest.payload;
            // allow ground above the camera for ground vehicles in valleys
            if (agl > -500 && agl < 5000)
            {
                heights.push_back(agl);
            }
        }
    }
    f.cameraTree.splitOutstanding();

    std::vector<double> nearestCameraDistances;
    nearestCameraDistances.reserve(cameraLocations.size());
    for (const auto &p : cameraLocations)
    {
        auto nn2 = f.cameraTree.searchKnn(toArray(p.topRows<2>()), 2);
        nearestCameraDistances.push_back(nn2.back().distance);
    }
    std::sort(nearestCameraDistances.begin(), nearestCameraDistances.end());

    f.medianCameraSpacing = std::sqrt(nearestCameraDistances[nearestCameraDistances.size() / 2]);

    if (heights.size() == 0)
        heights.push_back(f.medianCameraSpacing);
    std::sort(heights.begin(), heights.end());
    f.medianHeight = heights[heights.size() / 2];
    f.borderWidth = std::clamp(f.medianHeight * BORDER_PER_MEDIAN_HEIGHT, 0.0, 1000.0);
    return f;
}
} // namespace

namespace opencalibration
{

MeshGraph rebuildMesh(const point_cloud &cameraLocations, const std::vector<surface_model> &previousSurfaces)
{
    if (cameraLocations.size() < 2)
    {
        return MeshGraph();
    }

    const MeshFootprint footprint = computeFootprint(cameraLocations, previousSurfaces);
    const auto &vertexTree = footprint.vertexTree;
    const auto &cameraTree = footprint.cameraTree;
    const Eigen::Vector2d &cameraMin = footprint.cameraMin;
    const Eigen::Vector2d &cameraMax = footprint.cameraMax;
    const double medianHeight = footprint.medianHeight;
    const double minBorderWidth = footprint.borderWidth;

    const double minGridDistance = (cameraMax - cameraMin).norm() / 1000.0;
    double gridDistance = footprint.medianCameraSpacing;
    if (gridDistance < minGridDistance)
    {
        spdlog::debug("rebuildMesh: gridDistance {} too small, increasing to {}", gridDistance, minGridDistance);
        gridDistance = std::max(1e-3, minGridDistance);
    }

    MeshGraph newGraph;

    const double spanX = std::max(0., cameraMax.x() - cameraMin.x() + 2 * minBorderWidth);
    const double spanY = std::max(0., cameraMax.y() - cameraMin.y() + 2 * minBorderWidth);
    auto gridSize = [&](double span) { return static_cast<size_t>(std::ceil(span / gridDistance)) + 1; };

    constexpr size_t MAX_GRID_SIZE = 1000;
    if (gridSize(spanX) > MAX_GRID_SIZE || gridSize(spanY) > MAX_GRID_SIZE)
    {
        const double cappedGridDistance = std::max(spanX, spanY) / (MAX_GRID_SIZE - 1);
        spdlog::warn("Mesh grid too large: {}x{}, capping to {}. gridDistance: {} -> {}, medianHeight: {}",
                     gridSize(spanY), gridSize(spanX), MAX_GRID_SIZE, gridDistance, cappedGridDistance, medianHeight);
        gridDistance = cappedGridDistance;
    }
    const size_t rows = std::min(gridSize(spanY), MAX_GRID_SIZE);
    const size_t cols = std::min(gridSize(spanX), MAX_GRID_SIZE);

    spdlog::debug("Rebuilding mesh with {}x{} grid", rows, cols);

    Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> nodeIdGrid;
    nodeIdGrid.resize(rows, cols);
    for (size_t col = 0; col < cols; col++)
    {
        const double x = cameraMin.x() - minBorderWidth + gridDistance * col;

        for (size_t row = 0; row < rows; row++)
        {
            const double y = cameraMin.y() - minBorderWidth + gridDistance * row;

            const Eigen::Vector2d loc(x, y);
            const double z = vertexTree.size() > 0 ? vertexTree.search(toArray(loc)).payload
                                                   : cameraTree.search(toArray(loc)).payload - medianHeight;
            size_t nodeId = newGraph.addNode(MeshNode{Eigen::Vector3d(loc.x(), loc.y(), z)});
            nodeIdGrid(row, col) = nodeId;

            /*
             * Make edges with previous nodes: vertical, horizontal and diagonal
             *
             *  +  ------  +  ------  +  ------  +  ------  +
             *  |\         |\         |\         |\         |
             *  |  \       |  \       |  \       |  \       |
             *  |    \     |    \     |    \     |    \     |
             *  |      \   |      \   |      \   |      \   |
             *  |        \ |        \ |        \ |        \ |
             *  +  ------  +  ------  +  ------  +  ------  +
             *  |\         |\         |\         |\         |
             *  |  \       |  \       |  \       |  \       |
             *  |    \     |    \     |    \     |    \     |
             *  |      \   |      \   |      \   |      \   |
             *  |        \ |        \ |        \ |        \ |
             *  +  ------  +  ------  +  ------  +  ------  +
             *  |\         |\         |\         |\         |
             *  |  \       |  \       |  \       |  \       |
             *  |    \     |    \     |    \     |    \     |
             *  |      \   |      \   |      \   |      \   |
             *  |        \ |        \ |        \ |        \ |
             *  +  ------  +  ------  +  ------  +  ------  +             *
             */

            if (row > 0)
            {
                newGraph.addEdge({col == 0 || col + 1 == cols}, nodeId, nodeIdGrid(row - 1, col));
            }
            if (col > 0)
            {
                newGraph.addEdge({row == 0 || row + 1 == rows}, nodeId, nodeIdGrid(row, col - 1));
            }
            if (row > 0 && col > 0)
            {
                newGraph.addEdge({false}, nodeId, nodeIdGrid(row - 1, col - 1));
            }
        }
    }

    for (size_t col = 0; col < cols; col++)
    {
        for (size_t row = 0; row < rows; row++)
        {
            if (row > 0)
            {
                auto *edge = newGraph.getEdge(nodeIdGrid(row, col), nodeIdGrid(row - 1, col));
                if (col > 0)
                {
                    edge->payload.triangleOppositeNodes[0] = nodeIdGrid(row - 1, col - 1);
                }
                if (col + 1 < cols)
                {
                    edge->payload.triangleOppositeNodes[1] = nodeIdGrid(row, col + 1);
                    if (edge->payload.border)
                    {
                        std::swap(edge->payload.triangleOppositeNodes[0], edge->payload.triangleOppositeNodes[1]);
                    }
                }
            }
            if (col > 0)
            {
                auto *edge = newGraph.getEdge(nodeIdGrid(row, col), nodeIdGrid(row, col - 1));
                if (row > 0)
                {
                    edge->payload.triangleOppositeNodes[0] = nodeIdGrid(row - 1, col - 1);
                }
                if (row + 1 < rows)
                {
                    edge->payload.triangleOppositeNodes[1] = nodeIdGrid(row + 1, col);
                    if (edge->payload.border)
                    {
                        std::swap(edge->payload.triangleOppositeNodes[0], edge->payload.triangleOppositeNodes[1]);
                    }
                }
            }
            if (row > 0 && col > 0)
            {
                auto *edge = newGraph.getEdge(nodeIdGrid(row, col), nodeIdGrid(row - 1, col - 1));
                edge->payload.triangleOppositeNodes[0] = nodeIdGrid(row, col - 1);
                edge->payload.triangleOppositeNodes[1] = nodeIdGrid(row - 1, col);
            }
        }
    }

    return newGraph;
}

MeshGraph buildMinimalMesh(const point_cloud &cameraLocations, const std::vector<surface_model> &previousSurfaces)
{
    if (cameraLocations.size() < 2)
    {
        return MeshGraph();
    }

    const MeshFootprint footprint = computeFootprint(cameraLocations, previousSurfaces);
    const auto &vertexTree = footprint.vertexTree;
    const auto &cameraTree = footprint.cameraTree;
    const Eigen::Vector2d &cameraMin = footprint.cameraMin;
    const Eigen::Vector2d &cameraMax = footprint.cameraMax;
    const double medianHeight = footprint.medianHeight;
    const double minBorderWidth = footprint.borderWidth;

    const double xMin = cameraMin.x() - minBorderWidth;
    const double xMax = cameraMax.x() + minBorderWidth;
    const double yMin = cameraMin.y() - minBorderWidth;
    const double yMax = cameraMax.y() + minBorderWidth;

    auto getZ = [&](double x, double y) -> double {
        if (vertexTree.size() > 0)
        {
            std::vector<double> z;
            for (const auto &n : vertexTree.searchKnn(toArray(Eigen::Vector2d(x, y)), 32))
                z.push_back(n.payload);
            std::nth_element(z.begin(), z.begin() + z.size() / 2, z.end());
            return z[z.size() / 2];
        }
        return cameraTree.search(toArray(Eigen::Vector2d(x, y))).payload - medianHeight;
    };

    MeshGraph mesh;

    //   2 -------- 3
    //   | \        |
    //   |   \      |
    //   |     \    |
    //   |       \  |
    //   0 -------- 1
    //
    // Triangles: (0, 1, 3) and (0, 3, 2)

    size_t v0 = mesh.addNode(MeshNode{Eigen::Vector3d(xMin, yMin, getZ(xMin, yMin))});
    size_t v1 = mesh.addNode(MeshNode{Eigen::Vector3d(xMax, yMin, getZ(xMax, yMin))});
    size_t v2 = mesh.addNode(MeshNode{Eigen::Vector3d(xMin, yMax, getZ(xMin, yMax))});
    size_t v3 = mesh.addNode(MeshNode{Eigen::Vector3d(xMax, yMax, getZ(xMax, yMax))});

    auto addBorderEdge = [&mesh](size_t a, size_t b, size_t opposite) {
        MeshEdge edge;
        edge.border = true;
        edge.triangleOppositeNodes[0] = opposite;
        mesh.addEdge(edge, a, b);
    };
    addBorderEdge(v0, v1, v3);
    addBorderEdge(v1, v3, v0);
    addBorderEdge(v2, v3, v0);
    addBorderEdge(v0, v2, v3);

    MeshEdge diagEdge;
    diagEdge.border = false;
    diagEdge.triangleOppositeNodes = {v1, v2};
    mesh.addEdge(diagEdge, v0, v3);

    spdlog::info("Built minimal mesh with bounds [{}, {}] x [{}, {}]", xMin, xMax, yMin, yMax);

    return mesh;
}

} // namespace opencalibration
