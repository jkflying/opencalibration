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
                               const std::array<double, 2> &view = {})
{
    std::array<double, 3> residuals;
    RadiometricMatchCost{corr}(a.exposure, a.ab, a.slope, a.vig, b.exposure, b.ab, b.slope, b.vig, view.data(),
                               residuals.data());
    return residuals;
}

double viewLogCbrtGain(const SampleGeometry &g, const std::array<double, 2> &view)
{
    return view[0] * g.view_dir_x + view[1] * g.view_dir_y;
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
    // GIVEN: A point seen from two view directions with a global view-direction gain
    const std::array<float, 3> true_lab = {60, 0, 10};
    const std::array<double, 2> view = {0.1, -0.05};
    ColorCorrespondence corr = makeCorrespondence(1, 2, true_lab, true_lab);
    corr.geometry_a.view_dir_x = 0.4f;
    corr.geometry_b.view_dir_x = -0.05f;
    corr.geometry_b.view_dir_y = 0.1f;
    corr.lab_a = observeWithLogCbrtGain(true_lab, viewLogCbrtGain(corr.geometry_a, view));
    corr.lab_b = observeWithLogCbrtGain(true_lab, viewLogCbrtGain(corr.geometry_b, view));

    // WHEN: Evaluating without and with the view coefficients
    auto uncorrected = evaluate(corr, {}, {});
    auto corrected = evaluate(corr, {}, {}, view);

    // THEN: Only the corrected residual vanishes
    EXPECT_GT(std::abs(uncorrected[0]), 0.5);
    for (double r : corrected)
        EXPECT_NEAR(r, 0.0, 1e-4);
}

TEST(ColorBalance, shared_vignetting_cost_matches_separate_cost)
{
    // GIVEN: A correspondence between two images of the same camera model
    auto corr = makeCorrespondence(1, 2, {55, 3, 4}, {48, 2, 6});
    corr.geometry_a.normalized_radius = 0.9f;
    corr.geometry_b.view_dir_x = 0.3f;
    ImageParams a, b;
    a.exposure[0] = 0.02;
    a.vig[0] = b.vig[0] = -0.1;
    const std::array<double, 2> view = {-0.02, 0.03};

    // WHEN: Evaluating both cost variants
    auto separate = evaluate(corr, a, b, view);
    std::array<double, 3> shared;
    RadiometricMatchCostSharedVig{corr}(a.exposure, a.ab, a.slope, b.exposure, b.ab, b.slope, a.vig, view.data(),
                                        shared.data());

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

TEST(ColorBalance, weak_link_between_clusters_keeps_full_exposure_difference)
{
    // GIVEN: two well connected clusters of cameras, the second 0.2 brighter, joined by only a few correspondences
    std::vector<ColorCorrespondence> correspondences;
    const auto gain = [](size_t id) { return id >= 10 ? 0.2 : 0.0; };
    const auto link = [&](size_t a, size_t b, int n) {
        for (int i = 0; i < n; i++)
        {
            const std::array<float, 3> true_lab = {40.0f + (i % 30), 0.0f, 0.0f};
            correspondences.push_back(makeCorrespondence(a, b, observeWithLogCbrtGain(true_lab, gain(a)),
                                                         observeWithLogCbrtGain(true_lab, gain(b))));
        }
    };
    for (size_t c = 0; c < 20; c++)
        link(c, c / 10 * 10 + (c + 1) % 10, 100);
    link(0, 10, 5);

    // WHEN: we solve
    const auto result = solveColorBalance(correspondences);

    // THEN: the offset prior does not pull the clusters back towards their raw exposure
    ASSERT_TRUE(result.success);
    EXPECT_NEAR(result.per_image_params.at(15).log_cbrt_exposure - result.per_image_params.at(5).log_cbrt_exposure, 0.2,
                0.01);
}

TEST(ColorBalance, exif_exposure_anchors_chain_against_correspondence_bias)
{
    // GIVEN: a long chain of cameras with varying EXIF exposure, each link biased by 0.01 so the chain alone drifts
    // 0.89
    constexpr size_t n = 90;
    const auto exposure_value = [](size_t c) { return 1e-3 * (1.0 + 0.5 * static_cast<double>(c % 3)); };
    const auto gain = [&](size_t c) { return std::log(exposure_value(c) / exposure_value(1)) / 3.0; };
    ankerl::unordered_dense::map<size_t, double> exif_exposure_values;
    std::vector<ColorCorrespondence> correspondences;
    for (size_t c = 0; c < n; c++)
    {
        exif_exposure_values[c] = exposure_value(c);
        for (int i = 0; c + 1 < n && i < 20; i++)
        {
            const std::array<float, 3> true_lab = {40.0f + static_cast<float>(i % 30), 0.0f, 0.0f};
            correspondences.push_back(makeCorrespondence(c, c + 1, observeWithLogCbrtGain(true_lab, gain(c)),
                                                         observeWithLogCbrtGain(true_lab, gain(c + 1) + 0.01)));
        }
    }

    // WHEN: we solve with the EXIF exposures
    const auto result = solveColorBalance(correspondences, exif_exposure_values);

    // THEN: the drift stays bounded near the EXIF exposures instead of growing along the chain
    ASSERT_TRUE(result.success);
    EXPECT_NEAR(result.per_image_params.at(n - 1).log_cbrt_exposure - result.per_image_params.at(0).log_cbrt_exposure,
                gain(n - 1) - gain(0), 0.2);
    EXPECT_NEAR(result.per_image_params.at(n / 2).log_cbrt_exposure, gain(n / 2), 0.05);
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
    TargetPrior<3> prior{0.5};
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

TEST(ColorBalance, solve_recovers_horizontal_view_direction_gain)
{
    // GIVEN: a chain of overlapping cameras under a global horizontal view-direction gain
    constexpr size_t num_cameras = 30;
    const std::array<double, 2> true_view = {0.08, -0.05};
    std::vector<ColorCorrespondence> correspondences;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dir_dist(-0.6f, 0.6f);
    for (size_t c = 0; c < num_cameras; c++)
        for (size_t d = 1; d <= 3 && c + d < num_cameras; d++)
            for (int i = 0; i < 150; i++)
            {
                const std::array<float, 3> true_lab = {30.0f + static_cast<float>(i % 40), 5.0f, 0.0f};
                auto corr = makeCorrespondence(c, c + d, true_lab, true_lab);
                corr.geometry_a.view_dir_x = dir_dist(rng);
                corr.geometry_a.view_dir_y = dir_dist(rng);
                corr.geometry_b.view_dir_x = dir_dist(rng);
                corr.geometry_b.view_dir_y = dir_dist(rng);
                corr.lab_a = observeWithLogCbrtGain(true_lab, viewLogCbrtGain(corr.geometry_a, true_view));
                corr.lab_b = observeWithLogCbrtGain(true_lab, viewLogCbrtGain(corr.geometry_b, true_view));
                correspondences.push_back(corr);
            }

    // WHEN: we solve
    const auto result = solveColorBalance(correspondences);

    // THEN: the global horizontal view gain is recovered
    ASSERT_TRUE(result.success);
    EXPECT_NEAR(result.horizontal_view_dir_log_cbrt_gain[0], true_view[0], 0.01);
    EXPECT_NEAR(result.horizontal_view_dir_log_cbrt_gain[1], true_view[1], 0.01);
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
    double mean_exposure = 0;
    for (const auto &[cam_id, params] : result.per_image_params)
        mean_exposure += params.log_cbrt_exposure / result.per_image_params.size();
    EXPECT_NEAR(mean_exposure, 0.0, 0.003);
    EXPECT_NEAR(result.horizontal_view_dir_log_cbrt_gain[0], 0.0, 0.01);
    EXPECT_NEAR(result.horizontal_view_dir_log_cbrt_gain[1], 0.0, 0.01);
}
