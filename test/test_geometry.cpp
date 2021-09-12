
#include <gtest/gtest.h>

#include <opencalibration/geometry/KMeans.hpp>
#include <opencalibration/geometry/intersection.hpp>
#include <opencalibration/geometry/spectral_cluster.hpp>

using namespace opencalibration;

TEST(geometry, ray_intersection_nan_infinite_intersection)
{
    // GIVEN: just a single center and normal
    Eigen::Vector3d center(0, 0, 0), normal(0, 0, 1);

    // WHEN: there is an infinite intersection because it is with itself
    auto res = rayIntersection(ray_d{normal, center}, ray_d{normal, center});

    // THEN: the results should be NaN
    EXPECT_TRUE(res.first.array().isNaN().all());
    EXPECT_TRUE(std::isnan(res.second));
}

TEST(geometry, ray_intersection_never)
{
    // GIVEN: just a single normal, but 2 different points
    Eigen::Vector3d center1(0, 0, 0), center2(1, 0, 0), normal(0, 0, 1);

    // WHEN: there is no intersection because it is parallel forever
    auto res = rayIntersection(ray_d{normal, center1}, ray_d{normal, center2});

    // THEN: the results should be NaN
    EXPECT_TRUE(res.first.array().isNaN().all());
    EXPECT_TRUE(std::isnan(res.second));
}

TEST(geometry, ray_intersection_exact_at_origin)
{
    // GIVEN: two points that intersect at (0,0,0)
    Eigen::Vector3d point1(0, 0, -1), normal1(0, 0, 1);
    Eigen::Vector3d point2(0, 10, 0), normal2(0, 1, 0);

    // WHEN: there is a ray intersection
    auto res = rayIntersection(ray_d{normal1, point1}, ray_d{normal2, point2});

    // THEN: the results should be (0,0,0)
    EXPECT_DOUBLE_EQ((res.first - Eigen::Vector3d(0, 0, 0)).norm(), 0) << res.first.transpose();
    EXPECT_DOUBLE_EQ(res.second, 0);
}

TEST(geometry, ray_intersection_offset_from_origin)
{
    // GIVEN: two points that intersect at (1,0,0)
    Eigen::Vector3d point1(1, 0, 1), normal1(0, 0, 1);
    Eigen::Vector3d point2(1, 1, 0), normal2(0, 1, 0);

    // WHEN: there is a ray intersection
    auto res = rayIntersection(ray_d{normal1, point1}, ray_d{normal2, point2});

    // THEN: the results should be (1,0,0)
    EXPECT_DOUBLE_EQ((res.first - Eigen::Vector3d(1, 0, 0)).norm(), 0) << res.first.transpose();
    EXPECT_DOUBLE_EQ(res.second, 0);
}

TEST(geometry, ray_intersection_inexact)
{
    // GIVEN: two points that don't intersect at (1,0,0), but are on either side of it
    Eigen::Vector3d point1(2, 0, 1), normal1(0, 0, 1);
    Eigen::Vector3d point2(0, 1, 0), normal2(0, 1, 0);

    // WHEN: there is a ray intersection
    auto res = rayIntersection(ray_d{normal1, point1}, ray_d{normal2, point2});

    // THEN: the results should be (1,0,0), and with a distance of 2, behind the normals so negative
    EXPECT_DOUBLE_EQ((res.first - Eigen::Vector3d(1, 0, 0)).norm(), 0) << res.first.transpose();
    EXPECT_DOUBLE_EQ(res.second, -2 * 2);

    // WHEN: we reverse the normals and try again
    normal1 *= -1;
    normal2 *= -1;
    res = rayIntersection(ray_d{normal1, point1}, ray_d{normal2, point2});

    // THEN: the results should be (1,0,0), and with a distance of 2, in front of the normal so positive
    EXPECT_DOUBLE_EQ((res.first - Eigen::Vector3d(1, 0, 0)).norm(), 0) << res.first.transpose();
    EXPECT_DOUBLE_EQ(res.second, 2 * 2);
}

TEST(geometry, plane_conversion)
{
    // GIVEN: a plane defined by 3 points
    plane_3_corners_d p3;
    p3.corner[0] << 0, 0, -2;
    p3.corner[1] << 1, 0, -2;
    p3.corner[2] << 0, 1, -2;

    // WHEN: we convert it to a plane defined by a point and a normal
    plane_norm_offset pno = cornerPlane2normOffsetPlane(p3);

    // THEN: it should have the correct values

    EXPECT_NEAR(0, (Eigen::Vector3d(0, 0, -2) - pno.offset).norm(), 1e-9);
    EXPECT_NEAR(0, (Eigen::Vector3d(0, 0, 1) - pno.norm).norm(), 1e-9);
}

TEST(geometry, ray_plane_intersection)
{
    // GIVEN: a plane and a ray
    plane_norm_offset pno;
    pno.norm << 0, 0, 1;
    pno.offset << 5, 5, 2;

    ray_d r;
    r.dir << 0, 0, 0.5;
    r.offset << 3, 3, -10;

    // WHEN: we calculate the intersection
    Eigen::Vector3d i = rayPlaneIntersection(r, pno);

    // THEN: it should be where we expect
    EXPECT_NEAR(0, (Eigen::Vector3d(3, 3, 2) - i).norm(), 1e-9);
}

TEST(kmeans, clusters)
{
    opencalibration::KMeans<size_t, 2> kmeans(4);
    for (size_t i = 0; i < 50; i++)
        for (size_t j = 0; j < 20; j++)
            kmeans.add({(double)i, (double)j}, 20 * i + j);
    // print clusters
    auto print_clusters = [&]() {
        std::cout << "clusters: " << kmeans.getClusters().front().points.size() << " - "
                  << kmeans.getClusters().back().points.size() << " entries" << std::endl;
    };

    print_clusters();

    for (int i = 0; i < 12; i++)
    {
        kmeans.iterate();
        print_clusters();
    }
    print_clusters();
}

TEST(spectral, no_edges_no_spectralize)
{
    opencalibration::SpectralClustering<size_t, 2> spectral(4);
    for (size_t i = 0; i < 5; i++)
        for (size_t j = 0; j < 10; j++)
            spectral.add({(double)i, (double)j}, 10 * i + j);
    // print clusters
    auto print_clusters = [&]() {
        std::cout << spectral.getClusters().size() << " clusters: "
                  << "  sizes:" << spectral.getClusters().front().points.size() << " - "
                  << spectral.getClusters().back().points.size() << " entries" << std::endl;
    };

    print_clusters();

    EXPECT_FALSE(spectral.spectralize());

    spectral.fallback();

    for (int i = 0; i < 12; i++)
    {
        spectral.iterate();
        print_clusters();
    }
    print_clusters();
}

TEST(spectral, edges_spectralize)
{
    opencalibration::SpectralClustering<int, 2> spectral(4);
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            int id = 10 * i + j;
            spectral.add({(double)i, (double)j}, id);
            spectral.addLink(id, id + 1, 2);
            spectral.addLink(id, id + 9, 1);
            spectral.addLink(id, id + 10, 2);
            spectral.addLink(id, id + 11, 1);
        }
    }
    // print clusters
    auto print_clusters = [&]() {
        std::cout << spectral.getClusters().size() << " clusters: "
                  << "  sizes:" << spectral.getClusters().front().points.size() << " - "
                  << spectral.getClusters().back().points.size() << " entries" << std::endl;
    };

    print_clusters();

    EXPECT_TRUE(spectral.spectralize());

    for (int i = 0; i < 12; i++)
    {
        spectral.iterate();
        print_clusters();
    }
    print_clusters();
}
