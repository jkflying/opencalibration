#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/surface/intersect.hpp>
#include <opencalibration/types/mesh_graph.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

TEST(meshgraph, compiles)
{
    MeshGraph g;
}

TEST(meshgraph, expands_empty)
{
    MeshGraph g;
    point_cloud p;

    auto expanded = rebuildMesh(p, {surface_model{{}, g}});

    EXPECT_EQ(expanded.size_nodes(), 0);
    EXPECT_EQ(expanded.size_edges(), 0);
}

TEST(meshgraph, expands_single_point)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 0)};

    auto expanded = rebuildMesh(p, {surface_model{{}, g}});

    EXPECT_EQ(expanded.size_nodes(), 0);
    EXPECT_EQ(expanded.size_edges(), 0);
}

TEST(meshgraph, expands_2_points)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0)};

    auto expanded = rebuildMesh(p, {surface_model{{}, g}});

    EXPECT_EQ(expanded.size_nodes(), 30);
    EXPECT_EQ(expanded.size_edges(), 69);
}

TEST(meshgraph, intersects_rays)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    MeshIntersectionSearcher s;
    ASSERT_TRUE(s.init(g));

    for (int i = 0; i < 50; i++)
    {
        for (int j = 0; j < 50; j++)
        {
            const double x = -2 + j * (5. / 50);
            const double y = -2 + i * (4. / 50);

            const ray_d r{{0, 0, 1}, {x, y, 0}};
            const Eigen::Vector3d expectedIntersection(x, y, -1);
            auto intersection = s.triangleIntersect(r);

            EXPECT_EQ(intersection.type, MeshIntersectionSearcher::IntersectionInfo::INTERSECTION);
            EXPECT_LT((expectedIntersection - intersection.intersectionLocation).norm(), 1e-9)
                << intersection.intersectionLocation.transpose();
        }
    }
}

TEST(meshgraph, cycle_on_vertex_resolves)
{
    // Build a mesh with enough triangles that shooting a ray at a vertex
    // can cause the triangle walk to cycle between adjacent triangles.
    MeshGraph g;
    point_cloud p;
    for (double x = -2; x <= 2; x += 0.5)
    {
        for (double y = -2; y <= 2; y += 0.5)
        {
            p.push_back(Eigen::Vector3d(x, y, 0));
        }
    }
    g = rebuildMesh(p, {surface_model{{}, g}});

    ASSERT_GT(g.size_nodes(), 10);

    MeshIntersectionSearcher s;
    ASSERT_TRUE(s.init(g));

    // Shoot rays at every mesh vertex — these land exactly on shared edges
    // and are the classic trigger for walk oscillation
    for (auto it = g.cnodebegin(); it != g.cnodeend(); ++it)
    {
        const auto &loc = it->second.payload.location;
        const ray_d r{{0, 0, 1}, {loc.x(), loc.y(), loc.z() + 5}};
        auto result = s.triangleIntersect(r);

        EXPECT_NE(result.type, MeshIntersectionSearcher::IntersectionInfo::GRAPH_STRUCTURE_INCONSISTENT)
            << "Ray at vertex (" << loc.x() << ", " << loc.y() << ") should not fail";
        if (result.type == MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
        {
            EXPECT_NEAR(result.intersectionLocation.z(), loc.z(), 0.01);
        }
    }

    // Also shoot rays at edge midpoints
    for (auto it = g.cedgebegin(); it != g.cedgeend(); ++it)
    {
        const auto *src = g.getNode(it->second.getSource());
        const auto *dst = g.getNode(it->second.getDest());
        if (!src || !dst)
            continue;
        Eigen::Vector3d mid = (src->payload.location + dst->payload.location) * 0.5;
        const ray_d r{{0, 0, 1}, {mid.x(), mid.y(), mid.z() + 5}};
        auto result = s.triangleIntersect(r);

        EXPECT_NE(result.type, MeshIntersectionSearcher::IntersectionInfo::GRAPH_STRUCTURE_INCONSISTENT)
            << "Ray at edge midpoint (" << mid.x() << ", " << mid.y() << ") should not fail";
    }
}

TEST(meshgraph, doesnt_intersect_outside)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    MeshIntersectionSearcher s;
    ASSERT_TRUE(s.init(g));

    for (int i = 0; i < 50; i++)
    {
        const double x = -2.01;
        const double y = -2 + i * (4. / 50);

        const ray_d r{{0, 0, 1}, {x, y, 0}};
        auto intersection = s.triangleIntersect(r);
        EXPECT_EQ(intersection.type, MeshIntersectionSearcher::IntersectionInfo::OUTSIDE_BORDER);
    }

    for (int i = 0; i < 50; i++)
    {
        const double x = 3.01;
        const double y = -2 + i * (4. / 50);

        const ray_d r{{0, 0, 1}, {x, y, 0}};
        auto intersection = s.triangleIntersect(r);
        EXPECT_EQ(intersection.type, MeshIntersectionSearcher::IntersectionInfo::OUTSIDE_BORDER);
    }

    for (int i = 0; i < 50; i++)
    {
        const double x = -2 + i * (5. / 50);
        const double y = -2.01;

        const ray_d r{{0, 0, 1}, {x, y, 0}};
        auto intersection = s.triangleIntersect(r);
        EXPECT_EQ(intersection.type, MeshIntersectionSearcher::IntersectionInfo::OUTSIDE_BORDER);
    }

    for (int i = 0; i < 50; i++)
    {
        const double x = -2 + i * (5. / 50);
        const double y = 2.01;

        const ray_d r{{0, 0, 1}, {x, y, 0}};
        auto intersection = s.triangleIntersect(r);
        EXPECT_EQ(intersection.type, MeshIntersectionSearcher::IntersectionInfo::OUTSIDE_BORDER);
    }
}

TEST(meshgraph, long_walk_finds_containing_triangle)
{
    // GIVEN: a long flat mesh, with the searcher starting at the grid corner
    point_cloud p;
    for (int i = 0; i < 300; i++)
    {
        p.emplace_back(i, 0, 1);
    }
    MeshGraph g = rebuildMesh(p, {});
    MeshIntersectionSearcher s;
    ASSERT_TRUE(s.init(g));

    // WHEN: we intersect a ray far away from the start, needing more than 100 walk steps
    const ray_d r{{0, 0, 1}, {250.3, 0.2, 5}};
    const auto &result = s.triangleIntersect(r);

    // THEN: the result is an intersection on a triangle which contains the point
    ASSERT_EQ(result.type, MeshIntersectionSearcher::IntersectionInfo::INTERSECTION);
    EXPECT_GT(result.steps, 100u);
    Eigen::Vector2d lo = Eigen::Vector2d::Constant(INFINITY), hi = -lo;
    for (const auto *loc : result.nodeLocations)
    {
        lo = lo.cwiseMin(loc->topRows<2>());
        hi = hi.cwiseMax(loc->topRows<2>());
    }
    EXPECT_TRUE((lo.array() <= Eigen::Array2d(250.3, 0.2)).all() && (hi.array() >= Eigen::Array2d(250.3, 0.2)).all())
        << lo.transpose() << " / " << hi.transpose();
}

TEST(meshgraph, reinit_resumes_from_last_intersection)
{
    // GIVEN: a long flat mesh, with a searcher which found an intersection far from the grid corner then missed
    point_cloud p;
    for (int i = 0; i < 300; i++)
    {
        p.emplace_back(i, 0, 1);
    }
    MeshGraph g = rebuildMesh(p, {});
    MeshIntersectionSearcher s;
    ASSERT_TRUE(s.init(g));
    ASSERT_EQ(s.triangleIntersect({{0, 0, 1}, {250.3, 0.2, 5}}).type,
              MeshIntersectionSearcher::IntersectionInfo::INTERSECTION);
    ASSERT_NE(s.triangleIntersect({{0, 0, 1}, {250.3, 1e6, 5}}).type,
              MeshIntersectionSearcher::IntersectionInfo::INTERSECTION);

    // WHEN: we reinit and intersect near the last intersection
    ASSERT_TRUE(s.reinit());
    const auto &result = s.triangleIntersect({{0, 0, 1}, {251.3, 0.2, 5}});

    // THEN: the walk starts near the last intersection
    ASSERT_EQ(result.type, MeshIntersectionSearcher::IntersectionInfo::INTERSECTION);
    EXPECT_LT(result.steps, 10u);
}

TEST(meshgraph, capped_grid_covers_all_cameras)
{
    // GIVEN: cameras spread densely over a distance needing more than the maximum grid size
    point_cloud p;
    for (double x = 0; x <= 1000; x += 0.5)
    {
        p.emplace_back(x, 0, 1);
    }

    // WHEN: we build the mesh
    MeshGraph g = rebuildMesh(p, {});

    // THEN: the mesh still covers all of the cameras
    double minX = INFINITY, maxX = -INFINITY;
    for (auto it = g.cnodebegin(); it != g.cnodeend(); ++it)
    {
        minX = std::min(minX, it->second.payload.location.x());
        maxX = std::max(maxX, it->second.payload.location.x());
    }
    EXPECT_LE(minX, 0);
    EXPECT_GE(maxX, 1000);
}
