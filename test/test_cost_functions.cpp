#include <opencalibration/relax/relax_cost_function.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

TEST(cost_functions, difference_cost)
{
    DifferenceCost cost(2.0);
    double v1 = 5.0, v2 = 3.0;
    double residual = 0;
    EXPECT_TRUE(cost(&v1, &v2, &residual));
    EXPECT_DOUBLE_EQ(4.0, residual);
}

TEST(cost_functions, difference_cost_equal)
{
    DifferenceCost cost(1.0);
    double v = 7.0;
    double residual = 0;
    EXPECT_TRUE(cost(&v, &v, &residual));
    EXPECT_DOUBLE_EQ(0.0, residual);
}

TEST(cost_functions, distortion_monotonicity_zero_distortion)
{
    DistortionMonotonicityCost cost(1.0, 1.0);
    double radial[3] = {0, 0, 0};
    double residuals[10] = {};
    EXPECT_TRUE(cost(radial, residuals));
    for (int i = 0; i < 10; i++)
    {
        EXPECT_DOUBLE_EQ(0.0, residuals[i]) << "residual " << i;
    }
}

TEST(cost_functions, distortion_monotonicity_negative_k1)
{
    DistortionMonotonicityCost cost(1.0, 1.0);
    double radial[3] = {-10.0, 0, 0};
    double residuals[10] = {};
    EXPECT_TRUE(cost(radial, residuals));

    bool any_nonzero = false;
    for (int i = 0; i < 10; i++)
    {
        EXPECT_GE(residuals[i], 0.0);
        if (residuals[i] > 0)
            any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero);
}

TEST(cost_functions, adjacent_triangle_coplanar)
{
    // C and D on same side of AB produce coplanar normals
    Eigen::Vector2d A(0, 0), B(1, 0), C(0.5, 1), D(0.5, 0.5);
    AdjacentTriangleNormalCost cost(A, B, C, D, 1.0);

    double zA = 0, zB = 0, zC = 0, zD = 0;
    double residual = 0;
    EXPECT_TRUE(cost(&zA, &zB, &zC, &zD, &residual));
    EXPECT_NEAR(0.0, residual, 1e-5);
}

TEST(cost_functions, adjacent_triangle_noncoplanar)
{
    Eigen::Vector2d A(0, 0), B(1, 0), C(0.5, 1), D(0.5, -1);
    AdjacentTriangleNormalCost cost(A, B, C, D, 1.0);

    double zA = 0, zB = 0, zC = 0, zD = 5.0;
    double residual = 0;
    EXPECT_TRUE(cost(&zA, &zB, &zC, &zD, &residual));
    EXPECT_GT(std::abs(residual), 0.1);
}

TEST(cost_functions, robust_centroid_no_outlier)
{
    Eigen::Vector3d points[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Eigen::Vector3d result = robustCentroid(points, 3, 10.0);
    Eigen::Vector3d expected(1.0 / 3, 1.0 / 3, 1.0 / 3);
    EXPECT_NEAR(expected.x(), result.x(), 1e-6);
    EXPECT_NEAR(expected.y(), result.y(), 1e-6);
    EXPECT_NEAR(expected.z(), result.z(), 1e-6);
}

TEST(cost_functions, robust_centroid_with_outlier)
{
    Eigen::Vector3d points[4] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {100, 100, 100}};
    Eigen::Vector3d result = robustCentroid(points, 4, 1.0);

    Eigen::Vector3d naive_center(1.0 / 3, 1.0 / 3, 0);
    double dist_to_inliers = (result - naive_center).norm();
    double dist_to_outlier = (result - Eigen::Vector3d(100, 100, 100)).norm();
    EXPECT_LT(dist_to_inliers, dist_to_outlier);
}

TEST(cost_functions, angle_between_unit_vectors)
{
    Eigen::Vector3d a(1, 0, 0), b(0, 1, 0);
    double angle = angleBetweenUnitVectors(a, b);
    EXPECT_NEAR(M_PI / 2, angle, 1e-10);

    double zero_angle = angleBetweenUnitVectors(a, a);
    EXPECT_NEAR(0.0, zero_angle, 1e-5);
}
