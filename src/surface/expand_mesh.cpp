#include <opencalibration/surface/expand_mesh.hpp>

#include <jk/KDTree.h>
#include <spdlog/spdlog.h>

namespace
{
std::array<double, 2> toArray(const Eigen::Vector2d &vec)
{
    return {vec.x(), vec.y()};
}
} // namespace

namespace opencalibration
{

MeshGraph rebuildMesh(const point_cloud &cameraLocations, const std::vector<surface_model> &previousSurfaces)
{
    bool hasPreviousData = false;
    for (const auto &s : previousSurfaces)
    {
        if (s.mesh.size_nodes() > 0 || !s.cloud.empty())
        {
            hasPreviousData = true;
            break;
        }
    }

    if (cameraLocations.size() < 2 && !hasPreviousData)
    {
        return MeshGraph();
    }

    constexpr double HEIGHT_MARGIN =
        2; // make the mesh border twice as big as the average height of cameras above the mesh

    Eigen::Vector2d cameraMin = Eigen::Vector2d::Constant(std::numeric_limits<double>::max());
    Eigen::Vector2d cameraMax = -cameraMin;

    jk::tree::KDTree<double, 2> vertexTree, cameraTree;

    for (const auto &surface : previousSurfaces)
    {

        for (auto nodeIter = surface.mesh.cnodebegin(); nodeIter != surface.mesh.cnodeend(); ++nodeIter)
        {
            const Eigen::Vector3d p = nodeIter->second.payload.location;
            vertexTree.addPoint(toArray(p.topRows<2>()), p.z(), false);
        }
        for (const auto &cloud : surface.cloud)
        {
            for (const auto &p : cloud)
            {
                vertexTree.addPoint(toArray(p.topRows<2>()), p.z(), false);
            }
        }
    }
    vertexTree.splitOutstanding();

    std::vector<double> heights;
    heights.reserve(cameraLocations.size());

    double minCamZ = std::numeric_limits<double>::max();
    double maxCamZ = -minCamZ;

    for (const auto &p : cameraLocations)
    {
        cameraMin = cameraMin.cwiseMin(p.topRows<2>());
        cameraMax = cameraMax.cwiseMax(p.topRows<2>());
        minCamZ = std::min(minCamZ, p.z());
        maxCamZ = std::max(maxCamZ, p.z());
        cameraTree.addPoint(toArray(p.topRows<2>()), p.z(), false);

        if (vertexTree.size() > 0)
        {
            auto nearest = vertexTree.search(toArray(p.topRows<2>()));
            double agl = p.z() - nearest.payload;
            // Filter outlier heights: ground should be below camera but within reasonable distance.
            // We allow some range above camera for cases where it's a ground vehicle in a valley.
            if (agl > -500 && agl < 5000)
            {
                heights.push_back(agl);
            }
        }
    }
    cameraTree.splitOutstanding();

    std::vector<double> nearestCameraDistances;
    nearestCameraDistances.reserve(cameraLocations.size());
    for (const auto &p : cameraLocations)
    {
        auto nn2 = cameraTree.searchKnn(toArray(p.topRows<2>()), 2);
        nearestCameraDistances.push_back(nn2.back().distance);
    }
    std::sort(nearestCameraDistances.begin(), nearestCameraDistances.end());

    double gridDistance = nearestCameraDistances.size() < 2
                              ? std::numeric_limits<double>::infinity()
                              : std::sqrt(nearestCameraDistances[nearestCameraDistances.size() / 2]);

    // Ensure grid distance is not too small to avoid runaway grid density
    const double minGridDistance = (cameraMax - cameraMin).norm() / 1000.0;
    if (gridDistance < minGridDistance)
    {
        spdlog::debug("rebuildMesh: gridDistance {} too small, increasing to {}", gridDistance, minGridDistance);
        gridDistance = std::max(1e-3, minGridDistance);
    }

    if (heights.size() == 0)
        heights.push_back(std::isfinite(gridDistance) ? gridDistance : 10.0);
    std::sort(heights.begin(), heights.end());
    const double medianHeight = heights[heights.size() / 2];

    // Cap minBorderWidth to a reasonable value to avoid explosion if medianHeight is still wrong.
    // Also ensure it is non-negative.
    const double minBorderWidth = std::max(0.0, std::min(1000.0, medianHeight * HEIGHT_MARGIN));

    MeshGraph newGraph;

    size_t rows = static_cast<size_t>(
        std::ceil(std::max(0., cameraMax.y() - cameraMin.y() + 2 * minBorderWidth) / gridDistance)) + 1;
    size_t cols = static_cast<size_t>(
        std::ceil(std::max(0., cameraMax.x() - cameraMin.x() + 2 * minBorderWidth) / gridDistance)) + 1;

    if (rows > 1000 || cols > 1000)
    {
        spdlog::warn("Mesh grid too large: {}x{}, capping to 1000. gridDistance: {}, medianHeight: {}", rows, cols,
                     gridDistance, medianHeight);
        rows = std::min<size_t>(rows, 1000);
        cols = std::min<size_t>(cols, 1000);
    }

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

} // namespace opencalibration
