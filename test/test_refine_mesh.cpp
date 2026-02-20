#include <opencalibration/io/serialize.hpp>
#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/surface/intersect.hpp>
#include <opencalibration/surface/refine_mesh.hpp>
#include <opencalibration/types/mesh_graph.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>

using namespace opencalibration;

// Helper to check if two line segments intersect (not counting shared endpoints)
// Returns true if they cross each other in the interior
bool segmentsIntersect(const Eigen::Vector2d &p1, const Eigen::Vector2d &p2, const Eigen::Vector2d &p3,
                       const Eigen::Vector2d &p4)
{
    // Check if segments share an endpoint - that's allowed
    const double eps = 1e-10;
    if ((p1 - p3).squaredNorm() < eps || (p1 - p4).squaredNorm() < eps || (p2 - p3).squaredNorm() < eps ||
        (p2 - p4).squaredNorm() < eps)
    {
        return false;
    }

    // Compute cross products to determine orientation
    auto cross = [](const Eigen::Vector2d &o, const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
        return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x());
    };

    double d1 = cross(p3, p4, p1);
    double d2 = cross(p3, p4, p2);
    double d3 = cross(p1, p2, p3);
    double d4 = cross(p1, p2, p4);

    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)))
    {
        return true;
    }

    return false;
}

// Helper to validate mesh has no crossing edges
// Returns empty string if valid, or error message if invalid
std::string validateMeshNoCrossingEdges(const MeshGraph &mesh)
{
    // Collect all edges as 2D line segments
    std::vector<std::tuple<size_t, Eigen::Vector2d, Eigen::Vector2d>> edges;
    for (auto it = mesh.cedgebegin(); it != mesh.cedgeend(); ++it)
    {
        const auto &edge = it->second;
        const auto *srcNode = mesh.getNode(edge.getSource());
        const auto *dstNode = mesh.getNode(edge.getDest());
        if (srcNode && dstNode)
        {
            Eigen::Vector2d p1 = srcNode->payload.location.head<2>();
            Eigen::Vector2d p2 = dstNode->payload.location.head<2>();
            edges.emplace_back(it->first, p1, p2);
        }
    }

    // Check all pairs of edges for intersection
    for (size_t i = 0; i < edges.size(); ++i)
    {
        for (size_t j = i + 1; j < edges.size(); ++j)
        {
            const auto &[id1, p1, p2] = edges[i];
            const auto &[id2, p3, p4] = edges[j];

            if (segmentsIntersect(p1, p2, p3, p4))
            {
                std::stringstream ss;
                ss << "Edge " << id1 << " [(" << p1.x() << "," << p1.y() << ")->(" << p2.x() << "," << p2.y()
                   << ")] crosses edge " << id2 << " [(" << p3.x() << "," << p3.y() << ")->(" << p4.x() << "," << p4.y()
                   << ")]";
                return ss.str();
            }
        }
    }

    return "";
}

// Helper to check if a point lies on a line segment (not at endpoints)
bool pointOnSegment(const Eigen::Vector2d &p, const Eigen::Vector2d &a, const Eigen::Vector2d &b, double eps = 1e-9)
{
    // Check if p is at an endpoint - that's not a hanging node
    if ((p - a).squaredNorm() < eps || (p - b).squaredNorm() < eps)
    {
        return false;
    }

    // Check if p is collinear with a and b
    Eigen::Vector2d ab = b - a;
    Eigen::Vector2d ap = p - a;

    double cross = ab.x() * ap.y() - ab.y() * ap.x();
    if (std::abs(cross) > eps * ab.norm())
    {
        return false; // Not collinear
    }

    // Check if p is between a and b
    double dot = ap.dot(ab);
    double lenSq = ab.squaredNorm();

    return dot > eps && dot < lenSq - eps;
}

// Helper to validate mesh has no hanging nodes
// A hanging node is a vertex that lies on an edge but is not an endpoint
// Returns empty string if valid, or error message if invalid
std::string validateMeshNoHangingNodes(const MeshGraph &mesh)
{
    // Collect all vertices
    std::vector<std::pair<size_t, Eigen::Vector2d>> vertices;
    for (auto it = mesh.cnodebegin(); it != mesh.cnodeend(); ++it)
    {
        vertices.emplace_back(it->first, it->second.payload.location.head<2>());
    }

    // Collect all edges
    std::vector<std::tuple<size_t, size_t, size_t, Eigen::Vector2d, Eigen::Vector2d>> edges;
    for (auto it = mesh.cedgebegin(); it != mesh.cedgeend(); ++it)
    {
        const auto &edge = it->second;
        size_t srcId = edge.getSource();
        size_t dstId = edge.getDest();
        const auto *srcNode = mesh.getNode(srcId);
        const auto *dstNode = mesh.getNode(dstId);
        if (srcNode && dstNode)
        {
            Eigen::Vector2d p1 = srcNode->payload.location.head<2>();
            Eigen::Vector2d p2 = dstNode->payload.location.head<2>();
            edges.emplace_back(it->first, srcId, dstId, p1, p2);
        }
    }

    // Check each vertex against each edge
    for (const auto &[nodeId, nodePos] : vertices)
    {
        for (const auto &[edgeId, srcId, dstId, p1, p2] : edges)
        {
            // Skip if vertex is an endpoint of this edge
            if (nodeId == srcId || nodeId == dstId)
            {
                continue;
            }

            if (pointOnSegment(nodePos, p1, p2))
            {
                std::stringstream ss;
                ss << "Hanging node " << nodeId << " at (" << nodePos.x() << "," << nodePos.y() << ") lies on edge "
                   << edgeId << " [(" << p1.x() << "," << p1.y() << ")->(" << p2.x() << "," << p2.y() << ")]";
                return ss.str();
            }
        }
    }

    return "";
}

// Helper to count triangles in mesh
size_t countTriangles(const MeshGraph &mesh)
{
    std::set<std::array<size_t, 3>> triangles;
    for (auto it = mesh.cedgebegin(); it != mesh.cedgeend(); ++it)
    {
        const auto &edge = it->second;
        size_t src = edge.getSource();
        size_t dst = edge.getDest();

        // Side 0
        {
            std::array<size_t, 3> tri = {src, dst, edge.payload.triangleOppositeNodes[0]};
            std::sort(tri.begin(), tri.end());
            triangles.insert(tri);
        }

        // Side 1 if not border
        if (!edge.payload.border)
        {
            std::array<size_t, 3> tri = {src, dst, edge.payload.triangleOppositeNodes[1]};
            std::sort(tri.begin(), tri.end());
            triangles.insert(tri);
        }
    }
    return triangles.size();
}

TEST(refine_mesh, get_triangle_vertices)
{
    // Create a simple mesh
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(1, 1, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    ASSERT_GT(g.size_edges(), 0);

    // Get first edge and check triangle vertices
    auto it = g.cedgebegin();
    TriangleId tri{it->first, 0};

    auto verts = getTriangleVertices(g, tri);
    EXPECT_NE(verts[0], 0);
    EXPECT_NE(verts[1], 0);
    EXPECT_NE(verts[2], 0);

    // All vertices should be different
    EXPECT_NE(verts[0], verts[1]);
    EXPECT_NE(verts[1], verts[2]);
    EXPECT_NE(verts[0], verts[2]);
}

TEST(refine_mesh, find_longest_edge)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(1, 1, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    ASSERT_GT(g.size_edges(), 0);

    // Find a diagonal edge (which should be longest in isosceles right triangles)
    auto it = g.cedgebegin();
    TriangleId tri{it->first, 0};

    size_t longestId = findLongestEdge(g, tri);
    EXPECT_NE(longestId, 0);
}

TEST(refine_mesh, find_triangle_containing_point)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(2, 2, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Find triangle at center of mesh
    TriangleId tri = findTriangleContainingPoint(g, 1.0, 1.0);
    EXPECT_NE(tri.edgeId, 0);

    // Point outside mesh should return invalid
    TriangleId outside = findTriangleContainingPoint(g, 100.0, 100.0);
    EXPECT_EQ(outside.edgeId, 0);
}

TEST(refine_mesh, bisect_single_edge)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(2, 2, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    size_t initialNodes = g.size_nodes();
    size_t initialEdges = g.size_edges();
    size_t initialTriangles = countTriangles(g);

    // Find an interior edge (not border)
    size_t edgeToBisect = 0;
    for (auto it = g.cedgebegin(); it != g.cedgeend(); ++it)
    {
        if (!it->second.payload.border)
        {
            edgeToBisect = it->first;
            break;
        }
    }
    ASSERT_NE(edgeToBisect, 0);

    auto result = bisectEdge(g, edgeToBisect);

    EXPECT_NE(result.newVertexId, 0);
    EXPECT_EQ(result.splitEdgeIds.size(), 2);

    // Should have one more node
    EXPECT_EQ(g.size_nodes(), initialNodes + 1);

    // Edges: removed 1, added 2 (split) + 2 (to opposite corners) = +3
    EXPECT_EQ(g.size_edges(), initialEdges + 3);

    // Triangles: 2 original triangles become 4 (net +2), but due to mesh topology
    // the actual change might vary slightly
    EXPECT_GT(countTriangles(g), initialTriangles);

    // Validate no crossing edges
    std::string validationError = validateMeshNoCrossingEdges(g);
    EXPECT_EQ(validationError, "") << "Mesh has crossing edges after bisection: " << validationError;

    // Validate no hanging nodes
    std::string hangingError = validateMeshNoHangingNodes(g);
    EXPECT_EQ(hangingError, "") << "Mesh has hanging nodes after bisection: " << hangingError;
}

TEST(refine_mesh, refine_single_triangle)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(3, 3, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    size_t initialTriangles = countTriangles(g);

    // Find and refine a triangle at the center
    TriangleId tri = findTriangleContainingPoint(g, 1.5, 1.5);
    ASSERT_NE(tri.edgeId, 0);

    size_t created = refineTriangle(g, tri);
    EXPECT_GT(created, 0);

    // Should have more triangles now
    EXPECT_GT(countTriangles(g), initialTriangles);

    // Validate no crossing edges
    std::string validationError = validateMeshNoCrossingEdges(g);
    EXPECT_EQ(validationError, "") << "Mesh has crossing edges after refine: " << validationError;

    // Validate no hanging nodes
    std::string hangingError = validateMeshNoHangingNodes(g);
    EXPECT_EQ(hangingError, "") << "Mesh has hanging nodes after refine: " << hangingError;
}

TEST(refine_mesh, refine_at_point_single_level)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(4, 4, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    size_t initialNodes = g.size_nodes();

    size_t created = refineAtPoint(g, 2.0, 2.0, 1);
    EXPECT_GT(created, 0);
    EXPECT_GT(g.size_nodes(), initialNodes);

    // Validate no crossing edges
    std::string validationError = validateMeshNoCrossingEdges(g);
    EXPECT_EQ(validationError, "") << "Mesh has crossing edges after refine: " << validationError;

    // Validate no hanging nodes
    std::string hangingError = validateMeshNoHangingNodes(g);
    EXPECT_EQ(hangingError, "") << "Mesh has hanging nodes after refine: " << hangingError;
}

TEST(refine_mesh, refine_at_point_multiple_levels)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(4, 4, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    size_t initialNodes = g.size_nodes();

    // Refine 3 levels at center
    size_t created = refineAtPoint(g, 2.0, 2.0, 3);
    EXPECT_GT(created, 0);

    // Should have more nodes after multi-level refinement
    EXPECT_GT(g.size_nodes(), initialNodes);

    // Validate no crossing edges
    std::string validationError = validateMeshNoCrossingEdges(g);
    EXPECT_EQ(validationError, "") << "Mesh has crossing edges after multi-level refine: " << validationError;

    // Validate no hanging nodes
    std::string hangingError = validateMeshNoHangingNodes(g);
    EXPECT_EQ(hangingError, "") << "Mesh has hanging nodes after multi-level refine: " << hangingError;
}

TEST(refine_mesh, refine_where_circular_region)
{
    MeshGraph g;
    // Use a larger mesh area with closer cameras for better grid resolution
    point_cloud p{Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(10, 10, 1)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    size_t initialTriangles = countTriangles(g);

    // Refine triangles within radius 5.0 of center (5, 5) - large enough to catch some triangles
    double cx = 5.0, cy = 5.0, radius = 5.0;
    size_t created = refineWhere(
        g,
        [cx, cy, radius](double x, double y, double /*z*/) {
            double dx = x - cx;
            double dy = y - cy;
            return (dx * dx + dy * dy) < radius * radius;
        },
        3);

    EXPECT_GT(created, 0);
    EXPECT_GT(countTriangles(g), initialTriangles);

    // Validate no crossing edges
    std::string validationError = validateMeshNoCrossingEdges(g);
    EXPECT_EQ(validationError, "") << "Mesh has crossing edges after regional refine: " << validationError;

    // Validate no hanging nodes
    std::string hangingError = validateMeshNoHangingNodes(g);
    EXPECT_EQ(hangingError, "") << "Mesh has hanging nodes after regional refine: " << hangingError;
}

TEST(refine_mesh, original_mesh_no_crossing_edges)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(5, 5, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Original mesh should have no crossing edges
    std::string validationError = validateMeshNoCrossingEdges(g);
    EXPECT_EQ(validationError, "") << "Original mesh has crossing edges: " << validationError;

    // Original mesh should have no hanging nodes
    std::string hangingError = validateMeshNoHangingNodes(g);
    EXPECT_EQ(hangingError, "") << "Original mesh has hanging nodes: " << hangingError;
}

// Functional tests that output PLY files for visual inspection

TEST(refine_mesh, output_original_mesh_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(5, 5, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_original.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "Original mesh: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, " << countTriangles(g)
              << " triangles" << std::endl;
}

TEST(refine_mesh, output_single_refine_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(5, 5, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Refine at center once
    refineAtPoint(g, 2.5, 2.5, 1);

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_single.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "After single refine: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, "
              << countTriangles(g) << " triangles" << std::endl;
}

TEST(refine_mesh, output_multi_level_refine_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(5, 5, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Refine at center 4 times to see propagation
    refineAtPoint(g, 2.5, 2.5, 4);

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_multi_level.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "After 4-level refine: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, "
              << countTriangles(g) << " triangles" << std::endl;
}

TEST(refine_mesh, output_regional_refine_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(8, 8, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Refine a circular region
    double cx = 4.0, cy = 4.0, radius = 2.0;
    refineWhere(
        g,
        [cx, cy, radius](double x, double y, double /*z*/) {
            double dx = x - cx;
            double dy = y - cy;
            return (dx * dx + dy * dy) < radius * radius;
        },
        5);

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_regional.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "After regional refine: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, "
              << countTriangles(g) << " triangles" << std::endl;
}

TEST(refine_mesh, output_corner_refine_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(6, 6, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Refine near a corner to see how propagation works at boundaries
    refineAtPoint(g, 0.5, 0.5, 5);

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_corner.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "After corner refine: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, "
              << countTriangles(g) << " triangles" << std::endl;
}

TEST(refine_mesh, output_multiple_points_refine_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(10, 10, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Refine at multiple points to see combined propagation
    refineAtPoint(g, 2.0, 2.0, 3);
    refineAtPoint(g, 8.0, 8.0, 3);
    refineAtPoint(g, 5.0, 5.0, 4);

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_multiple_points.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "After multiple point refines: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, "
              << countTriangles(g) << " triangles" << std::endl;
}

TEST(refine_mesh, output_edge_refine_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(8, 8, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Refine along an edge to see linear propagation
    for (double x = 1.0; x < 7.0; x += 1.0)
    {
        refineAtPoint(g, x, 4.0, 2);
    }

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_edge.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "After edge refine: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, "
              << countTriangles(g) << " triangles" << std::endl;
}

// Non-disabled functional tests that output PLY files
TEST(refine_mesh_functional, output_original_mesh_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(5, 5, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_original.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "Original mesh: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, " << countTriangles(g)
              << " triangles" << std::endl;
}

TEST(refine_mesh_functional, output_single_refine_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(5, 5, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Refine at center once
    refineAtPoint(g, 2.5, 2.5, 1);

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_single.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "After single refine: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, "
              << countTriangles(g) << " triangles" << std::endl;

    // Validate mesh integrity
    std::string crossingError = validateMeshNoCrossingEdges(g);
    EXPECT_EQ(crossingError, "") << crossingError;
    std::string hangingError = validateMeshNoHangingNodes(g);
    EXPECT_EQ(hangingError, "") << hangingError;
}

TEST(refine_mesh_functional, output_multi_level_refine_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(5, 5, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Refine at center 4 times to see propagation
    refineAtPoint(g, 2.5, 2.5, 4);

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_multi_level.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "After 4-level refine: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, "
              << countTriangles(g) << " triangles" << std::endl;

    // Validate mesh integrity
    std::string crossingError = validateMeshNoCrossingEdges(g);
    EXPECT_EQ(crossingError, "") << crossingError;
    std::string hangingError = validateMeshNoHangingNodes(g);
    EXPECT_EQ(hangingError, "") << hangingError;
}

TEST(refine_mesh_functional, output_regional_refine_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(8, 8, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Refine a circular region
    double cx = 4.0, cy = 4.0, radius = 2.0;
    refineWhere(
        g,
        [cx, cy, radius](double x, double y, double /*z*/) {
            double dx = x - cx;
            double dy = y - cy;
            return (dx * dx + dy * dy) < radius * radius;
        },
        5);

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_regional.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "After regional refine: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, "
              << countTriangles(g) << " triangles" << std::endl;

    // Validate mesh integrity
    std::string crossingError = validateMeshNoCrossingEdges(g);
    EXPECT_EQ(crossingError, "") << crossingError;
    std::string hangingError = validateMeshNoHangingNodes(g);
    EXPECT_EQ(hangingError, "") << hangingError;
}

TEST(refine_mesh_functional, output_corner_refine_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(6, 6, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Refine near a corner to see how propagation works at boundaries
    refineAtPoint(g, 0.5, 0.5, 5);

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_corner.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "After corner refine: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, "
              << countTriangles(g) << " triangles" << std::endl;

    // Validate mesh integrity
    std::string crossingError = validateMeshNoCrossingEdges(g);
    EXPECT_EQ(crossingError, "") << crossingError;
    std::string hangingError = validateMeshNoHangingNodes(g);
    EXPECT_EQ(hangingError, "") << hangingError;
}

TEST(refine_mesh_functional, output_multiple_points_refine_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(10, 10, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Refine at multiple points to see combined propagation
    refineAtPoint(g, 2.0, 2.0, 3);
    refineAtPoint(g, 8.0, 8.0, 3);
    refineAtPoint(g, 5.0, 5.0, 4);

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_multiple_points.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "After multiple point refines: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, "
              << countTriangles(g) << " triangles" << std::endl;

    // Validate mesh integrity
    std::string crossingError = validateMeshNoCrossingEdges(g);
    EXPECT_EQ(crossingError, "") << crossingError;
    std::string hangingError = validateMeshNoHangingNodes(g);
    EXPECT_EQ(hangingError, "") << hangingError;
}

TEST(refine_mesh_functional, output_edge_refine_ply)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(8, 8, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Refine along an edge to see linear propagation
    for (double x = 1.0; x < 7.0; x += 1.0)
    {
        refineAtPoint(g, x, 4.0, 2);
    }

    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_edge.ply");
    EXPECT_TRUE(serialize(g, out));
    out.close();

    std::cout << "After edge refine: " << g.size_nodes() << " nodes, " << g.size_edges() << " edges, "
              << countTriangles(g) << " triangles" << std::endl;

    // Validate mesh integrity
    std::string crossingError = validateMeshNoCrossingEdges(g);
    EXPECT_EQ(crossingError, "") << crossingError;
    std::string hangingError = validateMeshNoHangingNodes(g);
    EXPECT_EQ(hangingError, "") << hangingError;
}

// Tests for minimal mesh and point density refinement

TEST(refine_mesh, build_minimal_mesh)
{
    point_cloud cameras;
    cameras.push_back(Eigen::Vector3d(0, 0, 10));
    cameras.push_back(Eigen::Vector3d(5, 0, 10));
    cameras.push_back(Eigen::Vector3d(5, 5, 10));
    cameras.push_back(Eigen::Vector3d(0, 5, 10));

    MeshGraph mesh = buildMinimalMesh(cameras, {});

    // Should have 4 nodes (corners) and 5 edges (4 border + 1 diagonal)
    EXPECT_EQ(mesh.size_nodes(), 4);
    EXPECT_EQ(mesh.size_edges(), 5);

    // Should have exactly 2 triangles
    EXPECT_EQ(countTriangles(mesh), 2);

    // Validate mesh integrity
    std::string crossingError = validateMeshNoCrossingEdges(mesh);
    EXPECT_EQ(crossingError, "") << crossingError;
    std::string hangingError = validateMeshNoHangingNodes(mesh);
    EXPECT_EQ(hangingError, "") << hangingError;
}

TEST(refine_mesh, minimal_mesh_ray_intersection_covers_full_square)
{
    // Create cameras at corners of a square
    point_cloud cameras;
    cameras.push_back(Eigen::Vector3d(0, 0, 10));
    cameras.push_back(Eigen::Vector3d(10, 0, 10));
    cameras.push_back(Eigen::Vector3d(10, 10, 10));
    cameras.push_back(Eigen::Vector3d(0, 10, 10));

    MeshGraph mesh = buildMinimalMesh(cameras, {});

    // Verify mesh structure
    ASSERT_EQ(mesh.size_nodes(), 4);
    ASSERT_EQ(mesh.size_edges(), 5);
    ASSERT_EQ(countTriangles(mesh), 2);

    // Print mesh structure for debugging
    std::cout << "Mesh nodes:" << std::endl;
    for (auto it = mesh.cnodebegin(); it != mesh.cnodeend(); ++it)
    {
        std::cout << "  Node " << it->first << ": " << it->second.payload.location.transpose() << std::endl;
    }

    std::cout << "Mesh edges:" << std::endl;
    for (auto it = mesh.cedgebegin(); it != mesh.cedgeend(); ++it)
    {
        std::cout << "  Edge " << it->first << ": " << it->second.getSource() << " -> " << it->second.getDest()
                  << " border=" << it->second.payload.border << " opp=[" << it->second.payload.triangleOppositeNodes[0]
                  << ", " << it->second.payload.triangleOppositeNodes[1] << "]" << std::endl;
    }

    // Initialize intersection searcher
    MeshIntersectionSearcher searcher;
    ASSERT_TRUE(searcher.init(mesh));

    // Test ray intersection across the entire mesh area
    // The mesh should cover approximately [-20, 30] x [-20, 30] with the border margin
    // But the actual mesh bounds depend on the height calculation
    // Let's test within the camera bounds which should definitely be covered

    int hitCount = 0;
    int missCount = 0;
    int totalTests = 0;

    // Test a grid of points across the mesh
    for (double x = 1; x <= 9; x += 1)
    {
        for (double y = 1; y <= 9; y += 1)
        {
            // Ray pointing downward from above
            const ray_d r{{0, 0, -1}, {x, y, 100}};
            auto intersection = searcher.triangleIntersect(r);

            totalTests++;
            if (intersection.type == MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
            {
                hitCount++;
                // Verify the intersection point is at the expected x,y
                EXPECT_NEAR(intersection.intersectionLocation.x(), x, 1e-6) << "At (" << x << ", " << y << ")";
                EXPECT_NEAR(intersection.intersectionLocation.y(), y, 1e-6) << "At (" << x << ", " << y << ")";
            }
            else
            {
                missCount++;
                std::cout << "MISS at (" << x << ", " << y << ") - type: " << static_cast<int>(intersection.type)
                          << std::endl;
            }
        }
    }

    std::cout << "Ray intersection results: " << hitCount << " hits, " << missCount << " misses out of " << totalTests
              << " tests" << std::endl;

    // All points within the camera bounds should intersect
    EXPECT_EQ(missCount, 0) << "Some rays within camera bounds did not intersect the mesh";
    EXPECT_EQ(hitCount, totalTests);

    // Specifically test points in both triangles:
    // Triangle 1 (bottom-right): contains point (7, 3)
    // Triangle 2 (top-left): contains point (3, 7)

    {
        const ray_d r{{0, 0, -1}, {7, 3, 100}};
        auto intersection = searcher.triangleIntersect(r);
        EXPECT_EQ(intersection.type, MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
            << "Point (7,3) should be in bottom-right triangle";
    }

    {
        const ray_d r{{0, 0, -1}, {3, 7, 100}};
        auto intersection = searcher.triangleIntersect(r);
        EXPECT_EQ(intersection.type, MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
            << "Point (3,7) should be in top-left triangle";
    }

    // Test diagonal - should be in the mesh
    {
        const ray_d r{{0, 0, -1}, {5, 5, 100}};
        auto intersection = searcher.triangleIntersect(r);
        EXPECT_EQ(intersection.type, MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
            << "Point (5,5) on diagonal should intersect";
    }
}

TEST(refine_mesh, minimal_mesh_not_reused_from_ground_plane_triangle)
{
    // This test verifies the fix for the bug where a 3-node triangle from GROUND_PLANE
    // was incorrectly reused when MINIMAL_MESH (4-node square) was requested.

    // Create a 3-node triangle mesh (simulating what GROUND_PLANE creates)
    MeshGraph triangleMesh;
    size_t v0 = triangleMesh.addNode(MeshNode{Eigen::Vector3d(-100, -100, 0)});
    size_t v1 = triangleMesh.addNode(MeshNode{Eigen::Vector3d(100, -100, 0)});
    size_t v2 = triangleMesh.addNode(MeshNode{Eigen::Vector3d(0, 100, 0)});
    triangleMesh.addEdge(MeshEdge{true, {v2, 0}}, v0, v1);
    triangleMesh.addEdge(MeshEdge{true, {v0, 0}}, v1, v2);
    triangleMesh.addEdge(MeshEdge{true, {v1, 0}}, v2, v0);

    ASSERT_EQ(triangleMesh.size_nodes(), 3);
    ASSERT_EQ(countTriangles(triangleMesh), 1);

    // Create camera locations for the minimal mesh
    point_cloud cameras;
    cameras.push_back(Eigen::Vector3d(0, 0, 10));
    cameras.push_back(Eigen::Vector3d(50, 0, 10));
    cameras.push_back(Eigen::Vector3d(50, 50, 10));
    cameras.push_back(Eigen::Vector3d(0, 50, 10));

    // Wrap the triangle in a surface_model as previousSurfaces
    std::vector<surface_model> previousSurfaces;
    previousSurfaces.push_back(surface_model{{}, triangleMesh});

    // Build minimal mesh - should NOT reuse the 3-node triangle
    MeshGraph minimalMesh = buildMinimalMesh(cameras, previousSurfaces);

    // The minimal mesh should have 4 nodes (square), not 3 (triangle)
    EXPECT_EQ(minimalMesh.size_nodes(), 4) << "Minimal mesh should have 4 nodes, not reuse 3-node triangle";
    EXPECT_EQ(minimalMesh.size_edges(), 5) << "Minimal mesh should have 5 edges";
    EXPECT_EQ(countTriangles(minimalMesh), 2) << "Minimal mesh should have 2 triangles";
}

TEST(refine_mesh, count_points_per_triangle)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(5, 5, 10)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // Create some test points
    std::vector<point_cloud> clouds;
    point_cloud testPoints;
    testPoints.push_back(Eigen::Vector3d(2.5, 2.5, 0)); // Center point
    testPoints.push_back(Eigen::Vector3d(1, 1, 0));
    testPoints.push_back(Eigen::Vector3d(4, 4, 0));
    clouds.push_back(testPoints);

    auto stats = countPointsPerTriangle(g, clouds);

    // Should have counted some points
    size_t totalCounted = 0;
    for (const auto &[key, s] : stats)
    {
        totalCounted += s.count;
    }
    EXPECT_GT(totalCounted, 0);
}

TEST(refine_mesh, refine_by_point_density)
{
    // Create a minimal mesh
    point_cloud cameras;
    cameras.push_back(Eigen::Vector3d(0, 0, 10));
    cameras.push_back(Eigen::Vector3d(10, 10, 10));

    MeshGraph mesh = buildMinimalMesh(cameras, {});
    ASSERT_EQ(mesh.size_nodes(), 4);

    // Create many points in the mesh to trigger refinement.
    // Points need varying z to produce non-zero distance variance.
    std::vector<point_cloud> clouds;
    point_cloud testPoints;
    for (double x = 1; x < 10; x += 0.5)
    {
        for (double y = 1; y < 10; y += 0.5)
        {
            double z = std::sin(x) * std::cos(y) * 2.0;
            testPoints.push_back(Eigen::Vector3d(x, y, z));
        }
    }
    clouds.push_back(testPoints);

    size_t initialNodes = mesh.size_nodes();
    size_t created = refineByPointDensity(mesh, clouds, 20, 0.0, 10);

    EXPECT_GT(created, 0);
    EXPECT_GT(mesh.size_nodes(), initialNodes);

    // Validate mesh integrity after refinement
    std::string crossingError = validateMeshNoCrossingEdges(mesh);
    EXPECT_EQ(crossingError, "") << crossingError;
    std::string hangingError = validateMeshNoHangingNodes(mesh);
    EXPECT_EQ(hangingError, "") << hangingError;

    std::cout << "After point density refinement: " << mesh.size_nodes() << " nodes, " << mesh.size_edges()
              << " edges, " << countTriangles(mesh) << " triangles" << std::endl;
}

TEST(refine_mesh, variance_filters_coplanar_points)
{
    MeshGraph mesh;
    size_t v0 = mesh.addNode(MeshNode{Eigen::Vector3d(0, 0, 0)});
    size_t v1 = mesh.addNode(MeshNode{Eigen::Vector3d(10, 0, 0)});
    size_t v2 = mesh.addNode(MeshNode{Eigen::Vector3d(0, 10, 0)});
    size_t v3 = mesh.addNode(MeshNode{Eigen::Vector3d(10, 10, 0)});

    size_t e01 = mesh.addEdge(MeshEdge{true, {0, 0}}, v0, v1);
    size_t e02 = mesh.addEdge(MeshEdge{true, {0, 0}}, v0, v2);
    size_t e13 = mesh.addEdge(MeshEdge{true, {0, 0}}, v1, v3);
    size_t e23 = mesh.addEdge(MeshEdge{true, {0, 0}}, v2, v3);
    size_t e03 = mesh.addEdge(MeshEdge{false, {0, 0}}, v0, v3);

    mesh.getEdge(e03)->payload.triangleOppositeNodes = {v1, v2};
    mesh.getEdge(e01)->payload.triangleOppositeNodes = {v3, 0};
    mesh.getEdge(e13)->payload.triangleOppositeNodes = {v0, 0};
    mesh.getEdge(e02)->payload.triangleOppositeNodes = {v3, 0};
    mesh.getEdge(e23)->payload.triangleOppositeNodes = {v0, 0};

    std::vector<point_cloud> clouds;
    point_cloud testPoints;
    for (double x = 0.5; x < 10; x += 0.3)
    {
        for (double y = 0.5; y < 10; y += 0.3)
        {
            testPoints.push_back(Eigen::Vector3d(x, y, 0));
        }
    }
    clouds.push_back(testPoints);

    auto stats = countPointsPerTriangle(mesh, clouds);
    for (const auto &[key, s] : stats)
    {
        EXPECT_GT(s.count, 20u);
        EXPECT_NEAR(s.distanceVariance, 0.0, 1e-10);
    }

    size_t initialNodes = mesh.size_nodes();
    size_t created = refineByPointDensity(mesh, clouds, 20, 0.01, 10);
    EXPECT_EQ(created, 0u);
    EXPECT_EQ(mesh.size_nodes(), initialNodes);
}

TEST(refine_mesh, variance_triggers_refinement_for_uneven_surface)
{
    MeshGraph mesh;
    size_t v0 = mesh.addNode(MeshNode{Eigen::Vector3d(0, 0, 0)});
    size_t v1 = mesh.addNode(MeshNode{Eigen::Vector3d(10, 0, 0)});
    size_t v2 = mesh.addNode(MeshNode{Eigen::Vector3d(0, 10, 0)});
    size_t v3 = mesh.addNode(MeshNode{Eigen::Vector3d(10, 10, 0)});

    size_t e01 = mesh.addEdge(MeshEdge{true, {0, 0}}, v0, v1);
    size_t e02 = mesh.addEdge(MeshEdge{true, {0, 0}}, v0, v2);
    size_t e13 = mesh.addEdge(MeshEdge{true, {0, 0}}, v1, v3);
    size_t e23 = mesh.addEdge(MeshEdge{true, {0, 0}}, v2, v3);
    size_t e03 = mesh.addEdge(MeshEdge{false, {0, 0}}, v0, v3);

    mesh.getEdge(e03)->payload.triangleOppositeNodes = {v1, v2};
    mesh.getEdge(e01)->payload.triangleOppositeNodes = {v3, 0};
    mesh.getEdge(e13)->payload.triangleOppositeNodes = {v0, 0};
    mesh.getEdge(e02)->payload.triangleOppositeNodes = {v3, 0};
    mesh.getEdge(e23)->payload.triangleOppositeNodes = {v0, 0};

    std::vector<point_cloud> clouds;
    point_cloud testPoints;
    for (double x = 0.5; x < 10; x += 0.3)
    {
        for (double y = 0.5; y < 10; y += 0.3)
        {
            double z = std::sin(x) * 0.5;
            testPoints.push_back(Eigen::Vector3d(x, y, z));
        }
    }
    clouds.push_back(testPoints);

    auto stats = countPointsPerTriangle(mesh, clouds);
    for (const auto &[key, s] : stats)
    {
        EXPECT_GT(s.count, 20u);
        EXPECT_GT(s.distanceVariance, 0.01);
    }

    size_t initialNodes = mesh.size_nodes();
    size_t created = refineByPointDensity(mesh, clouds, 20, 0.01, 10);
    EXPECT_GT(created, 0u);
    EXPECT_GT(mesh.size_nodes(), initialNodes);
}

TEST(refine_mesh_functional, output_minimal_mesh_refined_ply)
{
    // Create a minimal mesh and refine based on point density
    point_cloud cameras;
    cameras.push_back(Eigen::Vector3d(0, 0, 10));
    cameras.push_back(Eigen::Vector3d(10, 10, 10));

    MeshGraph mesh = buildMinimalMesh(cameras, {});

    // Create points to trigger refinement.
    // Points need varying z to produce non-zero distance variance.
    std::vector<point_cloud> clouds;
    point_cloud testPoints;
    for (double x = 0.5; x < 10; x += 0.3)
    {
        for (double y = 0.5; y < 10; y += 0.3)
        {
            double z = std::sin(x) * std::cos(y) * 2.0;
            testPoints.push_back(Eigen::Vector3d(x, y, z));
        }
    }
    clouds.push_back(testPoints);

    std::cout << "Initial minimal mesh: " << mesh.size_nodes() << " nodes, " << mesh.size_edges() << " edges, "
              << countTriangles(mesh) << " triangles, " << testPoints.size() << " points" << std::endl;

    // Refine until ~20 points per triangle
    refineByPointDensity(mesh, clouds, 20, 0.0, 20);

    std::cout << "After refinement: " << mesh.size_nodes() << " nodes, " << mesh.size_edges() << " edges, "
              << countTriangles(mesh) << " triangles" << std::endl;

    // Output PLY for inspection
    std::ofstream out(TEST_DATA_OUTPUT_DIR "refine_minimal_adaptive.ply");
    EXPECT_TRUE(serialize(mesh, out));
    out.close();

    // Validate mesh integrity
    std::string crossingError = validateMeshNoCrossingEdges(mesh);
    EXPECT_EQ(crossingError, "") << crossingError;
    std::string hangingError = validateMeshNoHangingNodes(mesh);
    EXPECT_EQ(hangingError, "") << hangingError;

    // Verify point counts are reasonable
    auto stats = countPointsPerTriangle(mesh, clouds);
    size_t maxCount = 0;
    for (const auto &[key, s] : stats)
    {
        maxCount = std::max(maxCount, s.count);
    }
    std::cout << "Max points per triangle after refinement: " << maxCount << std::endl;

    // Most triangles should have <= 20 points (allow some tolerance since refinement is iterative)
    EXPECT_LE(maxCount, 40) << "Expected refinement to reduce points per triangle";
}

TEST(refine_mesh, merge_surface_models)
{
    // Create a minimal mesh
    point_cloud cameras;
    cameras.push_back(Eigen::Vector3d(0, 0, 10));
    cameras.push_back(Eigen::Vector3d(10, 10, 10));

    MeshGraph baseMesh = buildMinimalMesh(cameras, {});

    // Get actual mesh bounds
    Eigen::Vector3d minBound = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    Eigen::Vector3d maxBound = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
    for (auto it = baseMesh.cnodebegin(); it != baseMesh.cnodeend(); ++it)
    {
        minBound = minBound.cwiseMin(it->second.payload.location);
        maxBound = maxBound.cwiseMax(it->second.payload.location);
    }

    // Create two surface models with the same mesh structure but different vertex Z values
    surface_model surf1, surf2;
    surf1.mesh = baseMesh;
    surf2.mesh = baseMesh;

    // Modify Z values in surf1 (shift up by 1)
    for (auto it = surf1.mesh.nodebegin(); it != surf1.mesh.nodeend(); ++it)
    {
        it->second.payload.location.z() += 1.0;
    }

    // Modify Z values in surf2 (shift down by 1)
    for (auto it = surf2.mesh.nodebegin(); it != surf2.mesh.nodeend(); ++it)
    {
        it->second.payload.location.z() -= 1.0;
    }

    // Create point clouds for each surface covering the mesh area
    // surf1 gets more points on the left, surf2 gets more points on the right
    double midX = (minBound.x() + maxBound.x()) / 2;
    point_cloud leftPoints, rightPoints;
    for (double x = minBound.x(); x < midX; x += 2)
    {
        for (double y = minBound.y(); y < maxBound.y(); y += 2)
        {
            leftPoints.push_back(Eigen::Vector3d(x, y, 0));
        }
    }
    for (double x = midX; x < maxBound.x(); x += 2)
    {
        for (double y = minBound.y(); y < maxBound.y(); y += 2)
        {
            rightPoints.push_back(Eigen::Vector3d(x, y, 0));
        }
    }
    surf1.cloud.push_back(leftPoints);
    surf2.cloud.push_back(rightPoints);

    std::cout << "surf1 has " << leftPoints.size() << " points, surf2 has " << rightPoints.size() << " points"
              << std::endl;

    // Merge the surfaces
    std::vector<surface_model> surfaces = {surf1, surf2};
    surface_model merged = mergeSurfaceModels(surfaces);

    // Verify merged mesh has same structure
    EXPECT_EQ(merged.mesh.size_nodes(), baseMesh.size_nodes());
    EXPECT_EQ(merged.mesh.size_edges(), baseMesh.size_edges());

    // Verify point clouds are combined
    EXPECT_EQ(merged.cloud.size(), 2);

    // Verify that merge happened (vertices should be different from both sources)
    // Compare merged Z values to base mesh Z values
    for (auto mergedIt = merged.mesh.cnodebegin(); mergedIt != merged.mesh.cnodeend(); ++mergedIt)
    {
        size_t nodeId = mergedIt->first;
        double mergedZ = mergedIt->second.payload.location.z();
        double surf1Z = surf1.mesh.getNode(nodeId)->payload.location.z();
        double surf2Z = surf2.mesh.getNode(nodeId)->payload.location.z();

        // At minimum, it should be within the range [surf2Z, surf1Z] (surf2 is lower, surf1 is higher)
        EXPECT_GE(mergedZ, surf2Z - 0.01) << "Merged Z below surf2 Z";
        EXPECT_LE(mergedZ, surf1Z + 0.01) << "Merged Z above surf1 Z";
    }

    std::cout << "Merged surface: " << merged.mesh.size_nodes() << " nodes, " << merged.mesh.size_edges() << " edges, "
              << merged.cloud.size() << " point clouds" << std::endl;
}

// Helper: check if (px, py) is inside the triangle defined by three 3D vertices (2D test, ignoring Z)
bool testPointInTriangle(const MeshGraph &mesh, const TriangleId &tri, double px, double py)
{
    auto verts = getTriangleVertices(mesh, tri);
    const auto *n0 = mesh.getNode(verts[0]);
    const auto *n1 = mesh.getNode(verts[1]);
    const auto *n2 = mesh.getNode(verts[2]);
    if (!n0 || !n1 || !n2)
        return false;

    auto sign = [](double p1x, double p1y, double p2x, double p2y, double p3x, double p3y) {
        return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y);
    };

    const auto &v0 = n0->payload.location;
    const auto &v1 = n1->payload.location;
    const auto &v2 = n2->payload.location;

    double d1 = sign(px, py, v0.x(), v0.y(), v1.x(), v1.y());
    double d2 = sign(px, py, v1.x(), v1.y(), v2.x(), v2.y());
    double d3 = sign(px, py, v2.x(), v2.y(), v0.x(), v0.y());

    bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(hasNeg && hasPos);
}

TEST(refine_mesh, triangle_locator_matches_brute_force_minimal_mesh)
{
    point_cloud cameras;
    cameras.push_back(Eigen::Vector3d(0, 0, 10));
    cameras.push_back(Eigen::Vector3d(10, 10, 10));
    MeshGraph mesh = buildMinimalMesh(cameras, {});

    TriangleLocator locator(mesh);

    for (double x = -15; x <= 25; x += 1.0)
    {
        for (double y = -15; y <= 25; y += 1.0)
        {
            TriangleId brute = findTriangleContainingPoint(mesh, x, y);
            TriangleId fast = locator.find(x, y);

            if (brute.edgeId == 0)
            {
                EXPECT_EQ(fast.edgeId, 0)
                    << "Locator found triangle at (" << x << ", " << y << ") but brute force didn't";
            }
            else
            {
                EXPECT_NE(fast.edgeId, 0) << "Locator missed triangle at (" << x << ", " << y << ")";
                // Point on a shared edge may map to either adjacent triangle — verify containment
                EXPECT_TRUE(testPointInTriangle(mesh, fast, x, y))
                    << "Locator returned non-containing triangle at (" << x << ", " << y << ")";
            }
        }
    }
}

TEST(refine_mesh, triangle_locator_matches_brute_force_grid_mesh)
{
    // Build a larger grid mesh with many triangles
    point_cloud cameras;
    cameras.push_back(Eigen::Vector3d(0, 0, 10));
    cameras.push_back(Eigen::Vector3d(10, 0, 10));
    cameras.push_back(Eigen::Vector3d(10, 10, 10));
    cameras.push_back(Eigen::Vector3d(0, 10, 10));
    MeshGraph mesh = rebuildMesh(cameras, {surface_model{{}, MeshGraph()}});

    ASSERT_GT(mesh.size_nodes(), 4);

    TriangleLocator locator(mesh);

    for (double x = -5; x <= 15; x += 0.7)
    {
        for (double y = -5; y <= 15; y += 0.7)
        {
            TriangleId brute = findTriangleContainingPoint(mesh, x, y);
            TriangleId fast = locator.find(x, y);

            if (brute.edgeId == 0)
            {
                EXPECT_EQ(fast.edgeId, 0) << "at (" << x << ", " << y << ")";
            }
            else
            {
                EXPECT_NE(fast.edgeId, 0) << "at (" << x << ", " << y << ")";
                EXPECT_TRUE(testPointInTriangle(mesh, fast, x, y)) << "at (" << x << ", " << y << ")";
            }
        }
    }
}

TEST(refine_mesh, triangle_locator_matches_brute_force_refined_mesh)
{
    // Build a mesh then refine it to create many small triangles
    point_cloud cameras;
    cameras.push_back(Eigen::Vector3d(0, 0, 10));
    cameras.push_back(Eigen::Vector3d(10, 10, 10));
    MeshGraph mesh = buildMinimalMesh(cameras, {});

    // Refine several times to create a dense mesh
    for (int i = 0; i < 5; i++)
    {
        refineAtPoint(mesh, 5, 5, 1);
    }
    refineAtPoint(mesh, 2, 2, 3);
    refineAtPoint(mesh, 8, 8, 3);

    ASSERT_GT(countTriangles(mesh), 10);

    TriangleLocator locator(mesh);

    for (double x = -15; x <= 25; x += 0.9)
    {
        for (double y = -15; y <= 25; y += 0.9)
        {
            TriangleId brute = findTriangleContainingPoint(mesh, x, y);
            TriangleId fast = locator.find(x, y);

            if (brute.edgeId == 0)
            {
                EXPECT_EQ(fast.edgeId, 0) << "at (" << x << ", " << y << ")";
            }
            else
            {
                EXPECT_NE(fast.edgeId, 0) << "at (" << x << ", " << y << ")";
                EXPECT_TRUE(testPointInTriangle(mesh, fast, x, y)) << "at (" << x << ", " << y << ")";
            }
        }
    }
}

TEST(refine_mesh, triangle_locator_empty_mesh)
{
    MeshGraph mesh;
    TriangleLocator locator(mesh);

    TriangleId result = locator.find(5.0, 5.0);
    EXPECT_EQ(result.edgeId, 0);
}
