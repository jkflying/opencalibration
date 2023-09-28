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
