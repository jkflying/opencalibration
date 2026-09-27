#include <opencalibration/ortho/color_balance.hpp>
#include <opencalibration/ortho/radiometric_cost.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using namespace opencalibration::orthomosaic;

namespace
{
struct ImageParams
{
    double exposure[1] = {0};
    double ab[2] = {0, 0};
    double brdf[1] = {0};
    double slope[2] = {0, 0};
    double vig[3] = {0, 0, 0};
};

std::array<float, 3> observeWithLogCbrtGain(std::array<float, 3> lab, double log_cbrt_gain)
{
    const double in[3] = {lab[0], lab[1], lab[2]};
    double out[3];
    scaleLabByCubeRootOfLinearGain(in, std::exp(log_cbrt_gain), out);
    return {static_cast<float>(out[0]), static_cast<float>(out[1]), static_cast<float>(out[2])};
}

std::array<double, 3> evaluate(const ColorCorrespondence &corr, const ImageParams &a, const ImageParams &b,
                               const double view_dir_gain[2] = nullptr)
{
    const double zero[2] = {0, 0};
    std::array<double, 3> residuals;
    RadiometricMatchCost{corr}(a.exposure, a.ab, a.brdf, a.slope, a.vig, b.exposure, b.ab, b.brdf, b.slope, b.vig,
                               view_dir_gain ? view_dir_gain : zero, residuals.data());
    return residuals;
}

ColorCorrespondence makeCorrespondence(size_t cam_a, size_t cam_b, std::array<float, 3> lab_a,
                                       std::array<float, 3> lab_b)
{
    ColorCorrespondence corr{};
    corr.lab_a = lab_a;
    corr.lab_b = lab_b;
    corr.camera_id_a = cam_a;
    corr.camera_id_b = cam_b;
    corr.model_id_a = 1;
    corr.model_id_b = 1;
    corr.geometry_a.normalized_radius = 0.3f;
    corr.geometry_b.normalized_radius = 0.3f;
    corr.geometry_a.view_angle_rad = 0.05f;
    corr.geometry_b.view_angle_rad = 0.05f;
    return corr;
}
} // namespace

TEST(ColorBalance, radiometric_match_cost_zero_residual)
{
    // GIVEN: Two cameras observing the same point with identical colors and zero corrections
    auto corr = makeCorrespondence(1, 2, {50, 10, -5}, {50, 10, -5});

    // WHEN: Evaluating the cost
    auto residuals = evaluate(corr, {}, {});

    // THEN: There is no residual
    for (double r : residuals)
        EXPECT_DOUBLE_EQ(r, 0.0);
}

TEST(ColorBalance, radiometric_match_cost_offset_difference)
{
    // GIVEN: Two cameras with different L channel values
    auto corr = makeCorrespondence(1, 2, {70, 0, 0}, {50, 0, 0});

    // WHEN: Evaluating with zero corrections
    auto residuals = evaluate(corr, {}, {});

    // THEN: The residual is the raw difference, expressed at mid-grey brightness
    const double mean_brightness_above_black = 60.0 + LAB_L_BLACK_OFFSET;
    EXPECT_DOUBLE_EQ(residuals[0], 20.0 * L_UNITS_PER_LOG_CBRT_GAIN / mean_brightness_above_black);
    EXPECT_DOUBLE_EQ(residuals[1], 0.0);
    EXPECT_DOUBLE_EQ(residuals[2], 0.0);
}

TEST(ColorBalance, radiometric_match_cost_exposure_gain_correction)
{
    // GIVEN: One colour seen by two cameras with different multiplicative exposure
    const std::array<float, 3> true_lab = {50, 20, -10};
    auto corr =
        makeCorrespondence(1, 2, observeWithLogCbrtGain(true_lab, 0.1), observeWithLogCbrtGain(true_lab, -0.05));
    ImageParams a, b;
    a.exposure[0] = 0.1;
    b.exposure[0] = -0.05;

    // WHEN: Evaluating with the matching exposure gains
    auto residuals = evaluate(corr, a, b);

    // THEN: L, a and b all agree
    for (double r : residuals)
        EXPECT_NEAR(r, 0.0, 1e-4);
}

TEST(ColorBalance, radiometric_match_cost_ab_offset_correction)
{
    // GIVEN: Two cameras with a white balance offset in a and b
    auto corr = makeCorrespondence(1, 2, {50, 13, -2}, {50, 10, -5});
    ImageParams a;
    a.ab[0] = 3;
    a.ab[1] = 3;

    // WHEN: Evaluating with the matching ab offset
    auto residuals = evaluate(corr, a, {});

    // THEN: There is no residual
    for (double r : residuals)
        EXPECT_NEAR(r, 0.0, 1e-9);
}

TEST(ColorBalance, radiometric_match_cost_slope_correction)
{
    // GIVEN: Directional brightness across the image, seen at opposite sides by two cameras
    const std::array<float, 3> true_lab = {40, 5, 5};
    const double slope_x = 0.2;
    auto corr = makeCorrespondence(1, 2, observeWithLogCbrtGain(true_lab, slope_x * 0.5),
                                   observeWithLogCbrtGain(true_lab, slope_x * -0.5));
    corr.geometry_a.normalized_x = 0.5f;
    corr.geometry_b.normalized_x = -0.5f;
    ImageParams a, b;
    a.slope[0] = slope_x;
    b.slope[0] = slope_x;

    // WHEN: Evaluating without and with the slope correction
    auto uncorrected = evaluate(corr, {}, {});
    auto corrected = evaluate(corr, a, b);

    // THEN: Only the corrected residual vanishes
    EXPECT_GT(std::abs(uncorrected[0]), 1.0);
    for (double r : corrected)
        EXPECT_NEAR(r, 0.0, 1e-4);
}

TEST(ColorBalance, radiometric_match_cost_view_direction_correction)
{
    // GIVEN: A point that looks brighter when viewed looking east than looking west
    const std::array<float, 3> true_lab = {60, 0, 10};
    const double view_dir_gain[2] = {0.3, 0.0};
    auto corr = makeCorrespondence(1, 2, observeWithLogCbrtGain(true_lab, 0.3 * 0.4),
                                   observeWithLogCbrtGain(true_lab, 0.3 * -0.4));
    corr.geometry_a.horizontal_view_dir_x = 0.4f;
    corr.geometry_b.horizontal_view_dir_x = -0.4f;

    // WHEN: Evaluating with the global view direction gain
    auto residuals = evaluate(corr, {}, {}, view_dir_gain);

    // THEN: There is no residual
    for (double r : residuals)
        EXPECT_NEAR(r, 0.0, 1e-4);
}

TEST(ColorBalance, shared_vignetting_cost_matches_separate_cost)
{
    // GIVEN: A correspondence between two images of the same camera model
    auto corr = makeCorrespondence(1, 2, {55, 3, 4}, {48, 2, 6});
    corr.geometry_a.normalized_radius = 0.9f;
    ImageParams a, b;
    a.exposure[0] = 0.02;
    b.brdf[0] = 0.1;
    a.vig[0] = b.vig[0] = -0.1;
    const double view_dir_gain[2] = {0.05, -0.02};

    // WHEN: Evaluating both cost variants
    auto separate = evaluate(corr, a, b, view_dir_gain);
    std::array<double, 3> shared;
    RadiometricMatchCostSharedVig{corr}(a.exposure, a.ab, a.brdf, a.slope, b.exposure, b.ab, b.brdf, b.slope, a.vig,
                                        view_dir_gain, shared.data());

    // THEN: They agree
    for (int c = 0; c < 3; c++)
        EXPECT_DOUBLE_EQ(shared[c], separate[c]);
}

TEST(ColorBalance, solve_synthetic_exposure_difference)
{
    // GIVEN: Two cameras with known multiplicative exposure differences
    std::vector<ColorCorrespondence> correspondences;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> radius_dist(0.0f, 1.0f);

    for (int i = 0; i < 200; i++)
    {
        const std::array<float, 3> true_lab = {30.0f + (i % 50), 5.0f, -5.0f};
        auto corr = makeCorrespondence(100, 200, observeWithLogCbrtGain(true_lab, 0.1),
                                       observeWithLogCbrtGain(true_lab, -0.05));
        corr.geometry_a.normalized_radius = radius_dist(rng);
        corr.geometry_b.normalized_radius = radius_dist(rng);
        correspondences.push_back(corr);
    }

    // WHEN: We solve color balance
    auto result = solveColorBalance(correspondences);

    // THEN: The solver converges with the relative exposure recovered
    EXPECT_TRUE(result.success);
    double relative =
        result.per_image_params.at(100).log_cbrt_exposure - result.per_image_params.at(200).log_cbrt_exposure;
    EXPECT_NEAR(relative, 0.15, 0.01);
}

TEST(ColorBalance, solve_three_cameras)
{
    // GIVEN: Three cameras forming a chain A-B, B-C with known exposure gains
    std::vector<ColorCorrespondence> correspondences;
    for (int i = 0; i < 100; i++)
    {
        const std::array<float, 3> true_lab = {40.0f + (i % 30), 0.0f, 0.0f};
        correspondences.push_back(
            makeCorrespondence(1, 2, observeWithLogCbrtGain(true_lab, 0.08), observeWithLogCbrtGain(true_lab, 0)));
        correspondences.push_back(
            makeCorrespondence(2, 3, observeWithLogCbrtGain(true_lab, 0.04), observeWithLogCbrtGain(true_lab, 0)));
    }

    // WHEN: We solve
    auto result = solveColorBalance(correspondences);

    // THEN: The chained relative exposures are recovered
    EXPECT_TRUE(result.success);
    double exp_a = result.per_image_params.at(1).log_cbrt_exposure;
    double exp_b = result.per_image_params.at(2).log_cbrt_exposure;
    double exp_c = result.per_image_params.at(3).log_cbrt_exposure;
    EXPECT_NEAR(exp_a - exp_b, 0.08, 0.015);
    EXPECT_NEAR(exp_b - exp_c, 0.04, 0.015);
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

TEST(ColorBalance, zero_prior_penalizes_params)
{
    // GIVEN: A prior with weight 0.5
    ZeroPrior<3> prior(0.5);
    double params[3] = {10.0, -5.0, 3.0};
    double residuals[3];

    // WHEN: Evaluating it
    prior(params, residuals);

    // THEN: Residuals are the weighted parameters
    EXPECT_DOUBLE_EQ(residuals[0], 5.0);
    EXPECT_DOUBLE_EQ(residuals[1], -2.5);
    EXPECT_DOUBLE_EQ(residuals[2], 1.5);
}

TEST(ColorBalance, remove_vignetting_undoes_linear_falloff_for_dark_and_bright)
{
    // GIVEN: A 40% linear-light falloff at the corner, applied to a dark and a bright colour
    auto xyz_to_lab = [](double X, double Y, double Z, double out[3]) {
        double fx = std::cbrt(X / 0.950456), fy = std::cbrt(Y), fz = std::cbrt(Z / 1.088754);
        out[0] = 116 * fy - 16;
        out[1] = 500 * (fx - fy);
        out[2] = 200 * (fy - fz);
    };
    const double corner_linear_falloff = 0.6;
    const double log_cbrt_falloff_coeffs[3] = {std::log(std::cbrt(corner_linear_falloff)), 0, 0};
    const float corner_radius = 1.0f;

    for (double luminance : {0.05, 0.6})
    {
        double X = 0.9 * luminance, Y = luminance, Z = 0.7 * luminance;
        double lab_true[3], lab_vignetted[3], lab_devignetted[3];
        xyz_to_lab(X, Y, Z, lab_true);
        xyz_to_lab(corner_linear_falloff * X, corner_linear_falloff * Y, corner_linear_falloff * Z, lab_vignetted);

        // WHEN: Removing vignetting with the matching coefficient
        removeVignetting(lab_vignetted, log_cbrt_falloff_coeffs, corner_radius, lab_devignetted);

        // THEN: The true colour is recovered regardless of brightness
        for (int c = 0; c < 3; c++)
            EXPECT_NEAR(lab_devignetted[c], lab_true[c], 1e-9) << "luminance " << luminance << " channel " << c;
    }
}

TEST(ColorBalance, solve_synthetic_directional_slope)
{
    // GIVEN: Camera A has a left-right brightness gradient, camera B has none
    std::vector<ColorCorrespondence> correspondences;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> pos_dist(-0.9f, 0.9f);
    const double true_slope_x = 0.1;

    for (int i = 0; i < 400; i++)
    {
        const std::array<float, 3> true_lab = {30.0f + (i % 40), 0.0f, 0.0f};
        float nx_a = pos_dist(rng);
        auto corr = makeCorrespondence(10, 20, observeWithLogCbrtGain(true_lab, true_slope_x * nx_a), true_lab);
        corr.geometry_a.normalized_x = nx_a;
        corr.geometry_a.normalized_y = pos_dist(rng);
        corr.geometry_b.normalized_x = pos_dist(rng);
        corr.geometry_b.normalized_y = pos_dist(rng);
        correspondences.push_back(corr);
    }

    // WHEN: We solve
    auto result = solveColorBalance(correspondences);

    // THEN: The relative x slope is recovered and there is no y slope
    EXPECT_TRUE(result.success);
    const auto &a = result.per_image_params.at(10);
    const auto &b = result.per_image_params.at(20);
    EXPECT_NEAR(a.slope[0] - b.slope[0], true_slope_x, 0.02);
    EXPECT_NEAR(a.slope[1], 0.0, 0.02);
    EXPECT_NEAR(b.slope[1], 0.0, 0.02);
}

TEST(ColorBalance, solve_synthetic_view_direction_gain)
{
    // GIVEN: Many cameras where every point is brighter when viewed looking towards the same bearing
    std::vector<ColorCorrespondence> correspondences;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dir_dist(-0.5f, 0.5f);
    std::uniform_int_distribution<size_t> cam_dist(0, 9);
    const double true_gain[2] = {0.08, -0.05};

    for (int i = 0; i < 2000; i++)
    {
        const std::array<float, 3> true_lab = {30.0f + (i % 40), 5.0f, 0.0f};
        size_t cam_a = cam_dist(rng), cam_b = (cam_a + 1 + cam_dist(rng) % 9) % 10;
        float vx_a = dir_dist(rng), vy_a = dir_dist(rng), vx_b = dir_dist(rng), vy_b = dir_dist(rng);
        auto corr = makeCorrespondence(cam_a, cam_b,
                                       observeWithLogCbrtGain(true_lab, true_gain[0] * vx_a + true_gain[1] * vy_a),
                                       observeWithLogCbrtGain(true_lab, true_gain[0] * vx_b + true_gain[1] * vy_b));
        corr.geometry_a.horizontal_view_dir_x = vx_a;
        corr.geometry_a.horizontal_view_dir_y = vy_a;
        corr.geometry_b.horizontal_view_dir_x = vx_b;
        corr.geometry_b.horizontal_view_dir_y = vy_b;
        correspondences.push_back(corr);
    }

    // WHEN: We solve
    auto result = solveColorBalance(correspondences);

    // THEN: The global view direction gain is recovered
    EXPECT_TRUE(result.success);
    EXPECT_NEAR(result.horizontal_view_dir_log_cbrt_gain[0], true_gain[0], 0.01);
    EXPECT_NEAR(result.horizontal_view_dir_log_cbrt_gain[1], true_gain[1], 0.01);
}

TEST(ColorBalance, solve_does_not_darken_to_shrink_noisy_residuals)
{
    // GIVEN: Correctly exposed cameras whose correspondences disagree only by unbiased noise
    std::vector<ColorCorrespondence> correspondences;
    std::mt19937 rng(3);
    std::normal_distribution<float> noise(0.0f, 8.0f);
    std::uniform_int_distribution<size_t> cam_dist(0, 9);

    for (int i = 0; i < 4000; i++)
    {
        const std::array<float, 3> true_lab = {30.0f + (i % 40), 5.0f, 0.0f};
        size_t cam_a = cam_dist(rng), cam_b = (cam_a + 1 + cam_dist(rng) % 9) % 10;
        correspondences.push_back(makeCorrespondence(
            cam_a, cam_b, {true_lab[0] + noise(rng), true_lab[1] + noise(rng), true_lab[2] + noise(rng)},
            {true_lab[0] + noise(rng), true_lab[1] + noise(rng), true_lab[2] + noise(rng)}));
    }

    // WHEN: We solve
    auto result = solveColorBalance(correspondences);

    // THEN: Cameras are not systematically darkened
    EXPECT_TRUE(result.success);
    double mean_exposure = 0, mean_brdf = 0;
    for (const auto &[cam_id, params] : result.per_image_params)
    {
        mean_exposure += params.log_cbrt_exposure / result.per_image_params.size();
        mean_brdf += params.brdf_coeff / result.per_image_params.size();
    }
    EXPECT_NEAR(mean_exposure, 0.0, 0.003);
    EXPECT_NEAR(mean_brdf, 0.0, 0.03);
}
