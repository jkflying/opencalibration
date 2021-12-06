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

    auto expanded = rebuildMesh(p, g);

    EXPECT_EQ(expanded.size_nodes(), 0);
    EXPECT_EQ(expanded.size_edges(), 0);
}

TEST(meshgraph, expands_single_point)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 0)};

    auto expanded = rebuildMesh(p, g);

    EXPECT_EQ(expanded.size_nodes(), 0);
    EXPECT_EQ(expanded.size_edges(), 0);
}

TEST(meshgraph, expands_2_points)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0)};

    auto expanded = rebuildMesh(p, g);

    EXPECT_EQ(expanded.size_nodes(), 30);
    EXPECT_EQ(expanded.size_edges(), 69);
}

TEST(meshgraph, intersects_ray)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0)};
    g = rebuildMesh(p, g);

    MeshIntersectionSearcher s;
    s.init(g);

    const ray_d r{{0, 0, 1}, {0.1, -0.1, 0}};


    const Eigen::Vector3d expectedIntersection (0.1, -0.1, -1);
    auto intersection = s.triangleIntersect(r);

    EXPECT_EQ(intersection.type, MeshIntersectionSearcher::IntersectionInfo::INTERSECTION);
    EXPECT_LT((expectedIntersection - intersection.intersectionLocation).norm(), 1e-9) << intersection.intersectionLocation.transpose();
}
