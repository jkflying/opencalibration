#pragma once

#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/mesh_graph.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <functional>
#include <vector>

namespace opencalibration
{

/**
 * @brief Result of a triangle bisection operation
 */
struct BisectionResult
{
    size_t newVertexId;               // ID of the newly created vertex at the midpoint
    size_t newEdgeId;                 // ID of the new edge connecting newVertex to the opposite corner
    std::vector<size_t> splitEdgeIds; // IDs of the two new edges replacing the bisected edge
};

/**
 * @brief Identifies a triangle by one of its edges and which side of the edge (0 or 1)
 */
struct TriangleId
{
    size_t edgeId;
    int side; // 0 or 1, indicating which triangleOppositeNode

    bool operator==(const TriangleId &other) const
    {
        return edgeId == other.edgeId && side == other.side;
    }
};

/**
 * @brief Hash function for TriangleId to use in unordered containers
 */
struct TriangleIdHash
{
    size_t operator()(const TriangleId &tri) const
    {
        // Combine edgeId and side using a simple hash combination
        return std::hash<size_t>()(tri.edgeId) ^ (std::hash<int>()(tri.side) << 1);
    }
};

/**
 * @brief Get the three vertex IDs of a triangle
 * @param mesh The mesh graph
 * @param tri The triangle identifier
 * @return Array of 3 vertex IDs forming the triangle, or empty array if invalid
 */
std::array<size_t, 3> getTriangleVertices(const MeshGraph &mesh, const TriangleId &tri);

/**
 * @brief Find the longest edge of a triangle (the edge to bisect in newest vertex bisection)
 *
 * For isosceles right triangles, this is the hypotenuse.
 *
 * @param mesh The mesh graph
 * @param tri The triangle identifier
 * @return Edge ID of the longest edge, or 0 if triangle is invalid
 */
size_t findLongestEdge(const MeshGraph &mesh, const TriangleId &tri);

/**
 * @brief Find a triangle containing the given point (x, y)
 * @param mesh The mesh graph
 * @param x X coordinate
 * @param y Y coordinate
 * @return TriangleId if found, or TriangleId with edgeId=0 if not found
 */
TriangleId findTriangleContainingPoint(const MeshGraph &mesh, double x, double y);

/**
 * @brief Bisect a single edge, splitting the triangles on both sides
 *
 * This is the core operation of newest vertex bisection. It:
 * 1. Creates a new vertex at the midpoint of the edge
 * 2. Splits the edge into two new edges
 * 3. Creates new edges from the midpoint to the opposite corners
 * 4. Updates all triangle connectivity
 *
 * @param mesh The mesh graph to modify
 * @param edgeId The edge to bisect
 * @return The bisection result, or empty result if edge not found
 */
BisectionResult bisectEdge(MeshGraph &mesh, size_t edgeId);

/**
 * @brief Refine a triangle using newest vertex bisection with conforming propagation
 *
 * This function bisects the triangle at its longest edge (hypotenuse for right triangles).
 * To maintain a conforming mesh (no hanging nodes), it recursively bisects neighbor
 * triangles as needed.
 *
 * The propagation follows the "newest vertex bisection" algorithm:
 * - Each triangle is bisected at its longest edge
 * - If the neighbor triangle shares that edge, both are split
 * - If the neighbor's longest edge is different, the neighbor is recursively refined first
 *
 * @param mesh The mesh graph to modify
 * @param tri The triangle to refine
 * @param maxDepth Maximum recursion depth to prevent runaway refinement (default 10)
 * @return Number of triangles created (2 per bisection)
 */
size_t refineTriangle(MeshGraph &mesh, const TriangleId &tri, int maxDepth = 10);

/**
 * @brief Refine mesh around a specific point
 *
 * Finds the triangle containing the point and refines it.
 *
 * @param mesh The mesh graph to modify
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param levels Number of refinement levels to apply
 * @return Number of triangles created
 */
size_t refineAtPoint(MeshGraph &mesh, double x, double y, int levels = 1);

/**
 * @brief Refine mesh in regions matching a predicate
 *
 * Iterates through all triangles and refines those where the predicate returns true.
 *
 * @param mesh The mesh graph to modify
 * @param shouldRefine Predicate taking triangle center (x, y, z) and returning true to refine
 * @param maxIterations Maximum number of refinement passes
 * @return Total number of triangles created
 */
size_t refineWhere(MeshGraph &mesh, std::function<bool(double x, double y, double z)> shouldRefine,
                   int maxIterations = 10);

/**
 * @brief Count the number of 3D points falling within each triangle
 *
 * @param mesh The mesh graph
 * @param points Vector of point clouds containing 3D points to count
 * @return Map from TriangleId to point count
 */
std::unordered_map<TriangleId, size_t, TriangleIdHash> countPointsPerTriangle(const MeshGraph &mesh,
                                                                              const std::vector<point_cloud> &points);

/**
 * @brief Refine triangles that have more than maxPointsPerTriangle points
 *
 * Uses newest vertex bisection to split triangles with too many points,
 * ensuring a conforming mesh (no hanging nodes).
 *
 * @param mesh The mesh graph to modify
 * @param points Vector of point clouds containing 3D points
 * @param maxPointsPerTriangle Maximum points allowed per triangle before refinement
 * @param maxIterations Maximum refinement iterations
 * @return Number of triangles created
 */
size_t refineByPointDensity(MeshGraph &mesh, const std::vector<point_cloud> &points, size_t maxPointsPerTriangle = 20,
                            int maxIterations = 20);

/**
 * @brief Legacy refineMesh function signature for compatibility
 */
void refineMesh(const MeasurementGraph &measurementGraph, MeshGraph &meshGraph);

} // namespace opencalibration
