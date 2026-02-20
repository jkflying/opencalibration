#include <opencalibration/ortho/color_balance.hpp>
#include <opencalibration/ortho/radiometric_cost.hpp>

#include <ceres/autodiff_cost_function.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>

using namespace opencalibration::orthomosaic;

TEST(ColorBalance, radiometric_match_cost_zero_residual)
{
    // GIVEN: Two cameras observing the same point with identical colors and zero corrections
    float lab_a[3] = {128, 128, 128};
    float lab_b[3] = {128, 128, 128};
    RadiometricMatchCost cost(lab_a, lab_b, 0.5f, 0.5f, 0.1f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f);

    double offset_a[3] = {0, 0, 0};
    double brdf_a[1] = {0};
    double vig_a[3] = {0, 0, 0};
    double offset_b[3] = {0, 0, 0};
    double brdf_b[1] = {0};
    double vig_b[3] = {0, 0, 0};
    double slope_a[2] = {0, 0};
    double slope_b[2] = {0, 0};
    double residuals[3];

    cost(offset_a, brdf_a, vig_a, offset_b, brdf_b, vig_b, slope_a, slope_b, residuals);

    EXPECT_DOUBLE_EQ(residuals[0], 0.0);
    EXPECT_DOUBLE_EQ(residuals[1], 0.0);
    EXPECT_DOUBLE_EQ(residuals[2], 0.0);
}

TEST(ColorBalance, radiometric_match_cost_offset_difference)
{
    // GIVEN: Two cameras with different L channel values
    float lab_a[3] = {150, 128, 128};
    float lab_b[3] = {130, 128, 128};
    RadiometricMatchCost cost(lab_a, lab_b, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    // With zero corrections, residual should be the difference
    double offset_a[3] = {0, 0, 0};
    double brdf_a[1] = {0};
    double vig_a[3] = {0, 0, 0};
    double offset_b[3] = {0, 0, 0};
    double brdf_b[1] = {0};
    double vig_b[3] = {0, 0, 0};
    double slope_a[2] = {0, 0};
    double slope_b[2] = {0, 0};
    double residuals[3];

    cost(offset_a, brdf_a, vig_a, offset_b, brdf_b, vig_b, slope_a, slope_b, residuals);

    EXPECT_DOUBLE_EQ(residuals[0], 20.0); // 150 - 130
    EXPECT_DOUBLE_EQ(residuals[1], 0.0);
    EXPECT_DOUBLE_EQ(residuals[2], 0.0);
}

TEST(ColorBalance, radiometric_match_cost_offset_correction)
{
    // GIVEN: Two cameras with L difference of 20, corrected by offsets
    float lab_a[3] = {150, 128, 128};
    float lab_b[3] = {130, 128, 128};
    RadiometricMatchCost cost(lab_a, lab_b, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    // Correct: offset_a = 10, offset_b = -10 -> (150-10) - (130+10) = 0
    double offset_a[3] = {10, 0, 0};
    double brdf_a[1] = {0};
    double vig_a[3] = {0, 0, 0};
    double offset_b[3] = {-10, 0, 0};
    double brdf_b[1] = {0};
    double vig_b[3] = {0, 0, 0};
    double slope_a[2] = {0, 0};
    double slope_b[2] = {0, 0};
    double residuals[3];

    cost(offset_a, brdf_a, vig_a, offset_b, brdf_b, vig_b, slope_a, slope_b, residuals);

    EXPECT_DOUBLE_EQ(residuals[0], 0.0);
}

TEST(ColorBalance, radiometric_match_cost_slope_zero_residual)
{
    // GIVEN: Two cameras with an L difference caused by directional brightness,
    // corrected by slope parameters
    float lab_a[3] = {130, 128, 128};
    float lab_b[3] = {120, 128, 128};
    // Image A: pixel at nx=0.5, ny=0.0; Image B: pixel at nx=-0.5, ny=0.0
    RadiometricMatchCost cost(lab_a, lab_b, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, -0.5f, 0.0f);

    double offset_a[3] = {0, 0, 0};
    double brdf_a[1] = {0};
    double vig_a[3] = {0, 0, 0};
    double offset_b[3] = {0, 0, 0};
    double brdf_b[1] = {0};
    double vig_b[3] = {0, 0, 0};
    // slope_x=10 for both: corr_a = 130 - 10*0.5 = 125, corr_b = 120 - 10*(-0.5) = 125
    double slope_a[2] = {10, 0};
    double slope_b[2] = {10, 0};
    double residuals[3];

    cost(offset_a, brdf_a, vig_a, offset_b, brdf_b, vig_b, slope_a, slope_b, residuals);

    EXPECT_NEAR(residuals[0], 0.0, 1e-10);
    EXPECT_DOUBLE_EQ(residuals[1], 0.0);
    EXPECT_DOUBLE_EQ(residuals[2], 0.0);
}

TEST(ColorBalance, radiometric_match_cost_slope_correction_effect)
{
    // GIVEN: Same colors at different image positions, slope should create a residual
    float lab_a[3] = {100, 128, 128};
    float lab_b[3] = {100, 128, 128};
    RadiometricMatchCost cost(lab_a, lab_b, 0.0f, 0.0f, 0.0f, 0.0f, 0.8f, 0.0f, -0.2f, 0.0f);

    double offset_a[3] = {0, 0, 0};
    double brdf_a[1] = {0};
    double vig_a[3] = {0, 0, 0};
    double offset_b[3] = {0, 0, 0};
    double brdf_b[1] = {0};
    double vig_b[3] = {0, 0, 0};
    double slope_a[2] = {5, 0};
    double slope_b[2] = {5, 0};
    double residuals[3];

    cost(offset_a, brdf_a, vig_a, offset_b, brdf_b, vig_b, slope_a, slope_b, residuals);

    // corr_a = 100 - 5*0.8 = 96, corr_b = 100 - 5*(-0.2) = 101, residual = -5
    EXPECT_NEAR(residuals[0], -5.0, 1e-6);
}

TEST(ColorBalance, solve_synthetic_exposure_difference)
{
    // GIVEN: Two cameras with a known exposure offset
    // Camera A has L offset of +10, Camera B has L offset of -5
    std::vector<ColorCorrespondence> correspondences;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> radius_dist(0.0f, 1.0f);

    for (int i = 0; i < 200; i++)
    {
        float true_L = 100.0f + (i % 50); // Varying scene luminance
        float true_a = 128.0f;
        float true_b = 128.0f;

        ColorCorrespondence corr;
        corr.lab_a = {true_L + 10.0f, true_a, true_b}; // Camera A sees brighter
        corr.lab_b = {true_L - 5.0f, true_a, true_b};  // Camera B sees darker
        corr.camera_id_a = 100;
        corr.camera_id_b = 200;
        corr.model_id_a = 1;
        corr.model_id_b = 1;
        corr.normalized_radius_a = radius_dist(rng);
        corr.normalized_radius_b = radius_dist(rng);
        corr.view_angle_a = 0.1f;
        corr.view_angle_b = 0.1f;
        correspondences.push_back(corr);
    }

    // WHEN: We solve color balance
    auto result = solveColorBalance(correspondences);

    // THEN: The solver should converge
    EXPECT_TRUE(result.success);

    // AND: The relative offset between cameras should be approximately 15
    // (10 - (-5) = 15) for the L channel
    double offset_a_L = result.per_image_params.at(100).lab_offset[0];
    double offset_b_L = result.per_image_params.at(200).lab_offset[0];
    double relative_offset = offset_a_L - offset_b_L;

    EXPECT_NEAR(relative_offset, 15.0, 1.0); // Allow some tolerance due to priors
}

TEST(ColorBalance, solve_three_cameras)
{
    // GIVEN: Three cameras forming a chain A-B, B-C with known offsets
    std::vector<ColorCorrespondence> correspondences;

    // A-B correspondences: A is 8 brighter than B
    for (int i = 0; i < 100; i++)
    {
        float true_L = 100.0f + (i % 30);
        ColorCorrespondence corr;
        corr.lab_a = {true_L + 8.0f, 128.0f, 128.0f};
        corr.lab_b = {true_L, 128.0f, 128.0f};
        corr.camera_id_a = 1;
        corr.camera_id_b = 2;
        corr.model_id_a = 1;
        corr.model_id_b = 1;
        corr.normalized_radius_a = 0.3f;
        corr.normalized_radius_b = 0.3f;
        corr.view_angle_a = 0.05f;
        corr.view_angle_b = 0.05f;
        correspondences.push_back(corr);
    }

    // B-C correspondences: B is 4 brighter than C
    for (int i = 0; i < 100; i++)
    {
        float true_L = 110.0f + (i % 30);
        ColorCorrespondence corr;
        corr.lab_a = {true_L + 4.0f, 128.0f, 128.0f};
        corr.lab_b = {true_L, 128.0f, 128.0f};
        corr.camera_id_a = 2;
        corr.camera_id_b = 3;
        corr.model_id_a = 1;
        corr.model_id_b = 1;
        corr.normalized_radius_a = 0.3f;
        corr.normalized_radius_b = 0.3f;
        corr.view_angle_a = 0.05f;
        corr.view_angle_b = 0.05f;
        correspondences.push_back(corr);
    }

    // WHEN: We solve
    auto result = solveColorBalance(correspondences);

    // THEN: Should converge
    EXPECT_TRUE(result.success);

    // AND: Relative offsets should be correct
    double off_a = result.per_image_params.at(1).lab_offset[0];
    double off_b = result.per_image_params.at(2).lab_offset[0];
    double off_c = result.per_image_params.at(3).lab_offset[0];

    EXPECT_NEAR(off_a - off_b, 8.0, 1.5);
    EXPECT_NEAR(off_b - off_c, 4.0, 1.5);
}

TEST(ColorBalance, solve_empty_correspondences)
{
    // GIVEN: No correspondences
    std::vector<ColorCorrespondence> empty;

    // WHEN: We solve
    auto result = solveColorBalance(empty);

    // THEN: Should report failure gracefully
    EXPECT_FALSE(result.success);
}

TEST(ColorBalance, exposure_prior_penalizes_offset)
{
    ExposurePrior prior(0.5);
    double offset[3] = {10.0, -5.0, 3.0};
    double residuals[3];

    prior(offset, residuals);

    EXPECT_DOUBLE_EQ(residuals[0], 5.0);
    EXPECT_DOUBLE_EQ(residuals[1], -2.5);
    EXPECT_DOUBLE_EQ(residuals[2], 1.5);
}

TEST(ColorBalance, vignetting_prior_penalizes_coeffs)
{
    VignettingPrior prior(0.1);
    double vig[3] = {1.0, 2.0, 3.0};
    double residuals[3];

    prior(vig, residuals);

    EXPECT_DOUBLE_EQ(residuals[0], 0.1);
    EXPECT_DOUBLE_EQ(residuals[1], 0.2);
    EXPECT_DOUBLE_EQ(residuals[2], 0.3);
}

TEST(ColorBalance, solve_synthetic_directional_slope)
{
    // GIVEN: Two cameras where one has a directional brightness gradient (left-right)
    // Camera A: slope_x = +8 (right side brighter), Camera B: no slope
    // Both see the same scene points from different image positions.
    std::vector<ColorCorrespondence> correspondences;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> nx_dist(-0.9f, 0.9f);
    std::uniform_real_distribution<float> ny_dist(-0.9f, 0.9f);

    const float true_slope_x = 8.0f;

    for (int i = 0; i < 400; i++)
    {
        float true_L = 80.0f + (i % 40);
        float nx_a = nx_dist(rng);
        float ny_a = ny_dist(rng);
        float nx_b = nx_dist(rng);
        float ny_b = ny_dist(rng);

        ColorCorrespondence corr;
        // Camera A observes L shifted by slope: observed = true + slope_x * nx
        corr.lab_a = {true_L + true_slope_x * nx_a, 128.0f, 128.0f};
        corr.lab_b = {true_L, 128.0f, 128.0f};
        corr.camera_id_a = 10;
        corr.camera_id_b = 20;
        corr.model_id_a = 1;
        corr.model_id_b = 1;
        corr.normalized_radius_a = 0.3f;
        corr.normalized_radius_b = 0.3f;
        corr.view_angle_a = 0.05f;
        corr.view_angle_b = 0.05f;
        corr.normalized_x_a = nx_a;
        corr.normalized_y_a = ny_a;
        corr.normalized_x_b = nx_b;
        corr.normalized_y_b = ny_b;
        correspondences.push_back(corr);
    }

    auto result = solveColorBalance(correspondences);
    EXPECT_TRUE(result.success);

    // Camera A should have a significant positive slope_x
    double slope_x_a = result.per_image_params.at(10).slope[0];
    double slope_y_a = result.per_image_params.at(10).slope[1];
    double slope_x_b = result.per_image_params.at(20).slope[0];
    double slope_y_b = result.per_image_params.at(20).slope[1];

    // The relative slope_x difference should approximate the true directional gradient
    EXPECT_NEAR(slope_x_a - slope_x_b, true_slope_x, 2.0);

    // Y slopes should be near zero (no vertical gradient in synthetic data)
    EXPECT_NEAR(slope_y_a, 0.0, 2.0);
    EXPECT_NEAR(slope_y_b, 0.0, 2.0);
}
