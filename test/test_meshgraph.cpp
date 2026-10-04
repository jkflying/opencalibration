#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/surface/intersect.hpp>
#include <opencalibration/types/mesh_graph.hpp>
#include <random>

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
    // GIVEN: a mesh with enough triangles that a ray at a vertex can make the triangle walk cycle
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

    Eigen::AlignedBox2d bounds;
    for (auto it = g.cnodebegin(); it != g.cnodeend(); ++it)
        bounds.extend(it->second.payload.location.topRows<2>());
    const auto expectResolved = [&](const Eigen::Vector3d &target) {
        const ray_d r{{0, 0, 1}, {target.x(), target.y(), target.z() + 5}};
        const auto result = s.triangleIntersect(r);
        const bool interior = target.x() > bounds.min().x() && target.x() < bounds.max().x() &&
                              target.y() > bounds.min().y() && target.y() < bounds.max().y();
        if (!interior)
        {
            EXPECT_NE(result.type, MeshIntersectionSearcher::IntersectionInfo::GRAPH_STRUCTURE_INCONSISTENT)
                << target.transpose();
            return;
        }
        EXPECT_EQ(result.type, MeshIntersectionSearcher::IntersectionInfo::INTERSECTION) << target.transpose();
        EXPECT_NEAR(result.intersectionLocation.z(), target.z(), 0.01) << target.transpose();
    };

    // WHEN: we shoot rays exactly at every vertex and edge midpoint, which lie on shared edges
    // THEN: every interior ray resolves to an intersection instead of cycling
    for (auto it = g.cnodebegin(); it != g.cnodeend(); ++it)
        expectResolved(it->second.payload.location);
    for (auto it = g.cedgebegin(); it != g.cedgeend(); ++it)
        expectResolved(
            (g.getNode(it->second.getSource())->payload.location + g.getNode(it->second.getDest())->payload.location) *
            0.5);
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

namespace
{
MeshGraph ridgeMesh(double ridgeHeight)
{
    surface_model ground;
    ground.cloud.push_back({Eigen::Vector3d(0, 0, 0)});
    MeshGraph g = rebuildMesh({Eigen::Vector3d(0, 0, 40), Eigen::Vector3d(1, 0, 40)}, {ground});
    for (auto it = g.nodebegin(); it != g.nodeend(); ++it)
    {
        Eigen::Vector3d &location = it->second.payload.location;
        location.z() = std::max(0.0, ridgeHeight * (1 - std::abs(location.x()) / 1.5));
    }
    return g;
}

MeshGraph roughMesh()
{
    MeshGraph g = ridgeMesh(0);
    for (auto it = g.nodebegin(); it != g.nodeend(); ++it)
    {
        Eigen::Vector3d &location = it->second.payload.location;
        location.z() =
            6 * std::sin(location.x() / 3) * std::cos(location.y() / 4) + (std::abs(location.x() - 5) < 2 ? 15 : 0);
    }
    return g;
}

bool hiddenByBruteForce(MeshLineOfSight &sight, const Eigen::Vector2d &xy, const Eigen::Vector3d &viewpoint,
                        double margin)
{
    const Eigen::Vector3d start(xy.x(), xy.y(), sight.surfaceHeight(xy));
    for (double t = 0.001; t < 1; t += 0.001)
    {
        const Eigen::Vector3d onSegment = start + t * (viewpoint - start);
        if (sight.surfaceHeight(onSegment.head<2>()) > onSegment.z() + margin)
            return true;
    }
    return false;
}
} // namespace

TEST(meshgraph, line_of_sight_over_flat_ground_is_clear)
{
    // GIVEN: a flat mesh
    MeshGraph g = ridgeMesh(0);
    MeshLineOfSight sight;
    ASSERT_TRUE(sight.init(g));

    // WHEN / THEN: ground points are visible from nadir and steeply oblique viewpoints
    for (double x = -20; x <= 20; x += 2.5)
    {
        EXPECT_TRUE(sight.surfaceVisibleFrom({x, 3}, {x, 3, 40})) << x;
        EXPECT_TRUE(sight.surfaceVisibleFrom({x, 3}, {x + 30, -10, 5})) << x;
    }
}

TEST(meshgraph, line_of_sight_blocked_by_ridge)
{
    // GIVEN: flat ground with a 30m ridge along x=0
    MeshGraph g = ridgeMesh(30);
    MeshLineOfSight sight;
    ASSERT_TRUE(sight.init(g));
    EXPECT_NEAR(sight.surfaceHeight({0, 2}), 30, 1e-9);
    EXPECT_NEAR(sight.surfaceHeight({4, 2}), 0, 1e-9);

    // WHEN / THEN: ground behind the ridge is hidden from the far side but visible from the near side
    EXPECT_FALSE(sight.surfaceVisibleFrom({4, 2}, {-15, 0, 40}));
    EXPECT_FALSE(sight.surfaceVisibleFrom({-4, 2}, {15, 0, 40}));
    EXPECT_TRUE(sight.surfaceVisibleFrom({4, 2}, {15, 0, 40}));
    EXPECT_TRUE(sight.surfaceVisibleFrom({-4, 2}, {-15, 0, 40}));

    // AND: the ridge top is visible from both sides
    EXPECT_TRUE(sight.surfaceVisibleFrom({0, 2}, {-15, 0, 40}));
    EXPECT_TRUE(sight.surfaceVisibleFrom({0, 2}, {15, 0, 40}));
}

TEST(meshgraph, line_of_sight_off_mesh_is_clear)
{
    // GIVEN: a mesh with a ridge, and an uninitialised line of sight
    MeshGraph g = ridgeMesh(30);
    MeshLineOfSight sight;
    MeshLineOfSight uninitialised;
    ASSERT_TRUE(sight.init(g));

    // WHEN / THEN: points outside the mesh, or without a mesh, are not treated as occluded
    EXPECT_TRUE(sight.surfaceVisibleFrom({1e4, 0}, {-15, 0, 40}));
    EXPECT_TRUE(uninitialised.surfaceVisibleFrom({4, 2}, {-15, 0, 40}));
}

TEST(meshgraph, line_of_sight_never_misses_occlusion_on_rough_terrain)
{
    // GIVEN: rolling terrain with a 15m cliff-sided wall, and random surface points and viewpoints over it
    MeshGraph g = roughMesh();
    MeshLineOfSight sight, reference;
    ASSERT_TRUE(sight.init(g));
    ASSERT_TRUE(reference.init(g));
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> horizontal(-20, 20), height(25, 80);

    // WHEN: each sight line is checked and compared to a dense brute-force march
    int hidden = 0, missed = 0, checked = 0;
    for (int i = 0; i < 2000; i++)
    {
        const Eigen::Vector2d xy(horizontal(rng), horizontal(rng));
        if (std::isnan(reference.surfaceHeight(xy)))
            continue;
        const Eigen::Vector3d viewpoint(horizontal(rng), horizontal(rng), height(rng));
        checked++;
        const bool clearlyHidden = hiddenByBruteForce(reference, xy, viewpoint, 0.5);
        hidden += clearlyHidden;
        missed += clearlyHidden && sight.surfaceVisibleFrom(xy, viewpoint);
    }

    // THEN: nothing clearly hidden is reported visible, and the scene does contain hidden points
    EXPECT_GT(checked, 1000);
    EXPECT_GT(hidden, 50);
    EXPECT_EQ(missed, 0);
}

TEST(meshgraph, line_of_sight_along_cell_boundaries_is_blocked)
{
    // GIVEN: a 10m ridge on a regular mesh, so mesh rows coincide with line-of-sight cell boundaries
    MeshGraph g = ridgeMesh(10);
    MeshLineOfSight sight;
    ASSERT_TRUE(sight.init(g));

    // WHEN / THEN: ground behind the ridge is hidden even when the sight line runs along a boundary
    for (double x : {3., 4.})
        for (double y : {0., 7.8575468949138947e-16, -7.8575468949138947e-16, 1.})
        {
            EXPECT_FALSE(sight.surfaceVisibleFrom({x, y}, {-15, 0, 40})) << x << "," << y;
            EXPECT_FALSE(sight.surfaceVisibleFrom({-x, y}, {15, 0, 40})) << x << "," << y;
            EXPECT_TRUE(sight.surfaceVisibleFrom({x, y}, {15, 0, 40})) << x << "," << y;
        }
}
