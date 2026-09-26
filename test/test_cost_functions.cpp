#include <opencalibration/distort/distort_keypoints.hpp>
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
    Eigen::Vector2d A(0, 0), B(1, 0), C(0.5, 1), D(0.5, -1);
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

TEST(cost_functions, plane_intersection_focal_radial_residual_not_reduced_by_larger_focal)
{
    InverseDifferentiableCameraModel<double> model;
    model.pixels_cols = 4000;
    model.pixels_rows = 3000;
    model.focal_length_pixels = 3000;
    model.principle_point << 2000, 1500;

    const Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    const Eigen::Vector3d cam0(0, 0, 100), cam1(20, 0, 100);
    std::array<double, 7> pose0{down.x(), down.y(), down.z(), down.w(), cam0.x(), cam0.y(), cam0.z()};
    std::array<double, 7> pose1{down.x(), down.y(), down.z(), down.w(), cam1.x(), cam1.y(), cam1.z()};
    const double z = 0;

    const Eigen::Vector2d pixel_error(5, 0);
    const auto residualNormForPixelError = [&](double focal, const Eigen::Vector3d &ground) {
        InverseDifferentiableCameraModel<double> scene_model = model;
        scene_model.focal_length_pixels = focal;
        const Eigen::Vector2d pixel0 = image_from_3d(down.inverse() * (ground - cam0), scene_model);
        const Eigen::Vector2d pixel1 = image_from_3d(down.inverse() * (ground - cam1), scene_model) + pixel_error;
        PlaneIntersectionAngleCost_OrientationFocalRadial_SharedModel cost(
            pixel0, pixel1, Eigen::Vector2d(-100, -100), Eigen::Vector2d(100, -100), Eigen::Vector2d(0, 100), model);
        std::array<double, 4> residuals;
        EXPECT_TRUE(cost(pose0.data(), pose1.data(), &z, &z, &z, &focal, model.principle_point.data(),
                         model.radial_distortion.data(), residuals.data()));
        return Eigen::Map<Eigen::Matrix<double, 4, 1>>(residuals.data()).norm();
    };

    const Eigen::Vector3d center(10, 5, 0), off_axis(150, 80, 0);
    EXPECT_GT(residualNormForPixelError(3000, center), 0);
    EXPECT_NEAR(residualNormForPixelError(6000, center) / residualNormForPixelError(3000, center), 1.0, 0.05);
    EXPECT_NEAR(residualNormForPixelError(6000, off_axis) / residualNormForPixelError(3000, off_axis), 1.0, 0.05);
    EXPECT_NEAR(residualNormForPixelError(3000, off_axis) / residualNormForPixelError(3000, center), 1.0, 0.1);
}
