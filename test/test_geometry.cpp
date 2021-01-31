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
