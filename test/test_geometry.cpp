#include <opencalibration/geometry/intersection.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

TEST(geometry, ray_intersection_nan_infinite_intersection)
{
    // GIVEN: just a single center and normal
    Eigen::Vector3d center(0, 0, 0), normal(0, 0, 1);

    // WHEN: there is an infinite intersection because it is with itself
    auto res = rayIntersection(center, normal, center, normal);

    // THEN: the results should be NaN
    EXPECT_TRUE(res.array().isNaN().all());
}

TEST(geometry, ray_intersection_never)
{
    // GIVEN: just a single normal, but 2 different points
    Eigen::Vector3d center1(0, 0, 0), center2(1, 0, 0), normal(0, 0, 1);

    // WHEN: there is no intersection because it is parallel forever
    auto res = rayIntersection(center1, normal, center2, normal);

    // THEN: the results should be NaN
    EXPECT_TRUE(res.array().isNaN().all());
}

TEST(geometry, ray_intersection_exact_at_origin)
{
    // GIVEN: two points that intersect at (0,0,0)
    Eigen::Vector3d point1(0, 0, -1), normal1(0, 0, 1);
    Eigen::Vector3d point2(0, 10, 0), normal2(0, 1, 0);

    // WHEN: there is a ray intersection
    auto res = rayIntersection(point1, normal1, point2, normal2);

    // THEN: the results should be (0,0,0)
    EXPECT_DOUBLE_EQ((res - Eigen::Vector4d(0, 0, 0, 0)).norm(), 0) << res.transpose();
}

TEST(geometry, ray_intersection_offset_from_origin)
{
    // GIVEN: two points that intersect at (1,0,0)
    Eigen::Vector3d point1(1, 0, 1), normal1(0, 0, 1);
    Eigen::Vector3d point2(1, 1, 0), normal2(0, 1, 0);

    // WHEN: there is a ray intersection
    auto res = rayIntersection(point1, normal1, point2, normal2);

    // THEN: the results should be (1,0,0)
    EXPECT_DOUBLE_EQ((res - Eigen::Vector4d(1, 0, 0, 0)).norm(), 0) << res.transpose();
}

TEST(geometry, ray_intersection_inexact)
{
    // GIVEN: two points that don't intersect at (1,0,0), but are on either side of it
    Eigen::Vector3d point1(2, 0, 1), normal1(0, 0, 1);
    Eigen::Vector3d point2(0, 1, 0), normal2(0, 1, 0);

    // WHEN: there is a ray intersection
    auto res = rayIntersection(point1, normal1, point2, normal2);

    // THEN: the results should be (1,0,0), and with a distance of 2
    EXPECT_DOUBLE_EQ((res.topRows<3>() - Eigen::Vector3d(1, 0, 0)).norm(), 0) << res.transpose();
    EXPECT_DOUBLE_EQ(res(3), 2 * 2);
}

TEST(geometry, plane_conversion)
{
    // GIVEN: a plane defined by 3 points
    plane_3_corners p3;
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

    ray r;
    r.dir << 0, 0, 0.5;
    r.offset << 3, 3, -10;

    // WHEN: we calculate the intersection
    Eigen::Vector3d i = rayPlaneIntersection(r, pno);

    // THEN: it should be where we expect
    EXPECT_NEAR(0, (Eigen::Vector3d(3, 3, 2) - i).norm(), 1e-9);
}
