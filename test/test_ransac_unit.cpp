#include <opencalibration/model_inliers/ransac.hpp>

#include <gtest/gtest.h>

#include <random>

using namespace opencalibration;

namespace
{
struct EssentialScene
{
    std::vector<correspondence> matches;
    Eigen::Matrix3d E;
};

// Camera 2 sees world point X at R * X + t, so x2^T [t]x R x1 = 0
EssentialScene makeEssentialScene(const std::vector<Eigen::Vector3d> &points)
{
    const Eigen::Matrix3d R = Eigen::AngleAxisd(0.2, Eigen::Vector3d(0.3, -1, 0.5).normalized()).toRotationMatrix();
    const Eigen::Vector3d t(1, 0.2, -0.3);
    Eigen::Matrix3d t_x;
    t_x << 0, -t.z(), t.y(), t.z(), 0, -t.x(), -t.y(), t.x(), 0;

    EssentialScene scene;
    scene.E = t_x * R;
    for (const auto &X : points)
        scene.matches.push_back(correspondence{X.normalized(), (R * X + t).normalized()});
    return scene;
}

Eigen::Vector3d scenePoint(size_t i, double depth_variation)
{
    const double a = static_cast<double>(i);
    return {std::sin(a * 1.7) * 3, std::cos(a * 2.3) * 3, 8 + std::sin(a * 0.9) * depth_variation};
}

EssentialScene makeEssentialScene(size_t n)
{
    std::vector<Eigen::Vector3d> points;
    for (size_t i = 0; i < n; i++)
        points.push_back(scenePoint(i, 2));
    return makeEssentialScene(points);
}

double essentialDistanceUpToScale(const Eigen::Matrix3d &a, const Eigen::Matrix3d &b)
{
    const Eigen::Matrix3d an = a / a.norm();
    const Eigen::Matrix3d bn = b / b.norm();
    return std::min((an - bn).norm(), (an + bn).norm());
}
} // namespace

TEST(ransac_homography, ransac_compiles)
{
    // GIVEN: some empty data
    std::vector<correspondence> matches;
    homography_model model;
    std::vector<bool> inliers;

    // WHEN: we get the ransac
    double score = ransac(matches, model, inliers);

    // THEN: it should have 0 score, no inliers
    EXPECT_EQ(score, 0);
    EXPECT_EQ(inliers.size(), 0);
}

TEST(ransac_homography, fits_identity)
{
    // GIVEN: 4 correspondences, from square A to square B when A == B
    std::vector<correspondence> matches;
    matches.push_back(correspondence{Eigen::Vector3d{1, 2, 1}, Eigen::Vector3d{1, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 2, 1}, Eigen::Vector3d{2, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 1, 1}, Eigen::Vector3d{2, 1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 1, 1}, Eigen::Vector3d{1, 1, 1}});

    homography_model model;
    std::vector<bool> inliers;

    // WHEN: we get the ransac model
    double score = ransac(matches, model, inliers);

    // THEN: it should have a 1 score because we only have 100% inliers
    EXPECT_DOUBLE_EQ(score, 1);

    // AND: the model should be correct, and all the points inliers, and the model an identity
    EXPECT_EQ(inliers.size(), 4);
    EXPECT_EQ(std::count(inliers.begin(), inliers.end(), true), 4);
    EXPECT_NEAR((model.homography - Eigen::Matrix3d::Identity()).norm(), 0, 1e-14);

    // AND: the decomposition should be an identity
    Eigen::Vector3d translation, translation2;
    Eigen::Quaterniond orientation, orientation2;
    std::array<decomposed_pose, 4> poses;
    ASSERT_TRUE(model.decompose(matches, inliers, poses));
    EXPECT_NEAR(poses[0].position.norm(), 0, 1e-14);
    EXPECT_NEAR(Eigen::AngleAxisd(poses[0].orientation).angle(), 0, 1e-14);
}

TEST(ransac_fundamental_matrix, ransac_compiles)
{
    // GIVEN: some empty data
    std::vector<correspondence> matches;
    fundamental_matrix_model model;
    std::vector<bool> inliers;

    // WHEN: we get the ransac
    double score = ransac(matches, model, inliers);

    // THEN: it should have 0 score, no inliers
    EXPECT_EQ(score, 0);
    EXPECT_EQ(inliers.size(), 0);
}

TEST(ransac_fundamental_matrix, fits_identity)
{
    // GIVEN: 4 correspondences, from square A to square B when A == B
    std::vector<correspondence> matches;
    matches.push_back(correspondence{Eigen::Vector3d{1, 2, 1}, Eigen::Vector3d{1, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 2, 1}, Eigen::Vector3d{2, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 1, 1}, Eigen::Vector3d{2, 1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 1, 1}, Eigen::Vector3d{1, 1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 2, 3}, Eigen::Vector3d{1, 2, 3}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 2, 2}, Eigen::Vector3d{2, 2, 2}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 1, 3}, Eigen::Vector3d{2, 1, 3}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 1, 2}, Eigen::Vector3d{1, 1, 2}});
    for (auto &m : matches)
    {
        m.measurement1.normalize();
        m.measurement2.normalize();
    }

    fundamental_matrix_model model;
    std::vector<bool> inliers;

    // WHEN: we get the ransac model
    double score = ransac(matches, model, inliers);

    // THEN: it should have a 1 score because we only have 100% inliers
    EXPECT_DOUBLE_EQ(score, 1);

    // AND: the model should be correct, and all the points inliers, and the model an identity
    EXPECT_EQ(inliers.size(), 8);
    EXPECT_EQ(std::count(inliers.begin(), inliers.end(), true), 8);
    EXPECT_NEAR(model.fundamental_matrix.norm(), 1, 1e-14) << model.fundamental_matrix;

    double total_error = 0;
    for (const auto &m : matches)
    {
        total_error += model.error(m);
    }

    EXPECT_NEAR(total_error, 0, 1e-10);
}

class ransac_p : public ::testing::TestWithParam<std::tuple<Eigen::Quaterniond, Eigen::Vector3d>>
{
};

TEST_P(ransac_p, homography_rotation_translation)
{
    // GIVEN: 4 correspondences, from square A to square B when A + (1,1,0) == rot(90) * B

    Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    Eigen::Quaterniond R = std::get<0>(GetParam());
    Eigen::Vector3d T = std::get<1>(GetParam());

    auto perspective = [](Eigen::Vector3d v, Eigen::Quaterniond R, Eigen::Vector3d T) -> Eigen::Vector3d {
        Eigen::Vector3d cam_loc(0, 0, 10);

        Eigen::Vector3d ray = R.inverse() * (v - (cam_loc + T));
        Eigen::Vector2d pixel = ray.hnormalized() * 600;

        return pixel.homogeneous();
    };

    std::vector<correspondence> matches;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            Eigen::Vector3d p(i > 0 ? -1 : 1, j > 0 ? -1 : 1, 0);
            Eigen::Vector3d pers = perspective(p, down, Eigen::Vector3d::Zero()), pers_rt = perspective(p, R * down, T);
            matches.push_back(correspondence{pers_rt, pers});
        }
    }

    homography_model model;
    std::vector<bool> inliers;

    // WHEN: we get the ransac model
    double score = ransac(matches, model, inliers);

    // THEN: it should have a 1 score because we only have 100% inliers
    EXPECT_DOUBLE_EQ(score, 1);

    // AND: the model should be correct, and all the points inliers, and the model as expected
    EXPECT_EQ(inliers.size(), 4);
    EXPECT_EQ(std::count(inliers.begin(), inliers.end(), true), 4);

    // AND: the decomposition should be what was input
    std::array<decomposed_pose, 4> poses;
    ASSERT_TRUE(model.decompose(matches, inliers, poses));

    double T_err[4];
    double R_err[4];
    double min_err = INFINITY;
    for (size_t i = 0; i < poses.size(); i++)
    {
        T_err[i] = (down * poses[i].position.normalized() - T.normalized()).norm();
        R_err[i] = Eigen::AngleAxisd((down * poses[i].orientation * down.inverse()).inverse() * R).angle();
        min_err = std::min(min_err, T_err[i] + R_err[i]);
    }

    EXPECT_NEAR(min_err, 0, 1e-7);

    if (::testing::Test::HasFailure())
    {
        std::cout << "R: " << R.coeffs().transpose() << "    T: " << T.transpose() << "   H:" << std::endl;
        std::cout << model.homography << std::endl;
    }
}

TEST(ransac_fundamental_matrix, fitInliers_uses_correct_subset)
{
    std::vector<correspondence> matches;
    matches.push_back(correspondence{Eigen::Vector3d{1, 2, 1}, Eigen::Vector3d{1, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{100, 200, 1}, Eigen::Vector3d{200, 100, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 2, 1}, Eigen::Vector3d{2, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{150, 250, 1}, Eigen::Vector3d{250, 150, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 1, 1}, Eigen::Vector3d{2, 1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 1, 1}, Eigen::Vector3d{1, 1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1.5, 1.5, 1}, Eigen::Vector3d{1.5, 1.5, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{120, 220, 1}, Eigen::Vector3d{220, 120, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1.2, 1.8, 1}, Eigen::Vector3d{1.2, 1.8, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{130, 230, 1}, Eigen::Vector3d{230, 130, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1.8, 1.2, 1}, Eigen::Vector3d{1.8, 1.2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1.3, 1.7, 1}, Eigen::Vector3d{1.3, 1.7, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 2, 3}, Eigen::Vector3d{1, 2, 3}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 2, 2}, Eigen::Vector3d{2, 2, 2}});
    for (auto &m : matches)
    {
        m.measurement1.normalize();
        m.measurement2.normalize();
    }

    std::vector<bool> inliers = {true,  false, true,  false, true, true, true,
                                 false, true,  false, true,  true, true, true};

    fundamental_matrix_model model;
    model.fitInliers(matches, inliers);

    double inlier_error = 0;
    size_t inlier_count = 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (inliers[i])
        {
            inlier_error += std::abs(model.error(matches[i]));
            inlier_count++;
        }
    }
    double avg_inlier_error = inlier_error / inlier_count;

    double outlier_error = 0;
    size_t outlier_count = 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (!inliers[i])
        {
            outlier_error += std::abs(model.error(matches[i]));
            outlier_count++;
        }
    }
    double avg_outlier_error = outlier_error / outlier_count;

    EXPECT_LT(avg_inlier_error, 0.01);
    EXPECT_GT(avg_outlier_error, avg_inlier_error * 2);
}

TEST(ransac_fundamental_matrix, evaluate_uses_absolute_error)
{
    std::vector<correspondence> matches;
    matches.push_back(correspondence{Eigen::Vector3d{1, 2, 1}, Eigen::Vector3d{1, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 2, 1}, Eigen::Vector3d{2, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 1, 1}, Eigen::Vector3d{2, 1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 1, 1}, Eigen::Vector3d{1, 1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1.5, 1.5, 1}, Eigen::Vector3d{1.5, 1.5, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1.2, 1.8, 1}, Eigen::Vector3d{1.2, 1.8, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1.8, 1.2, 1}, Eigen::Vector3d{1.8, 1.2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1.3, 1.7, 1}, Eigen::Vector3d{1.3, 1.7, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 2, 3}, Eigen::Vector3d{1, 2, 3}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 2, 2}, Eigen::Vector3d{2, 2, 2}});
    for (auto &m : matches)
    {
        m.measurement1.normalize();
        m.measurement2.normalize();
    }

    fundamental_matrix_model model;
    std::vector<bool> inliers;

    double score = ransac(matches, model, inliers);

    EXPECT_GT(score, 0.7);
    EXPECT_GE(std::count(inliers.begin(), inliers.end(), true), 8);
}

TEST(ransac_iterations, required_iterations_follow_confidence_formula)
{
    // GIVEN/WHEN/THEN: 50% inliers with 4-point samples needs log(0.001) / log(1 - 0.5^4) iterations
    EXPECT_EQ(ransacIterationsForConfidence(0.5, 4), 107);
    EXPECT_EQ(ransacIterationsForConfidence(1.0, 4), 20);
}

TEST(ransac_iterations, tiny_inlier_ratio_saturates_at_maximum)
{
    // GIVEN: inlier ratios so small that 1 - ratio^n rounds to 1
    // WHEN/THEN: the iteration count saturates instead of converting inf to an integer
    EXPECT_EQ(ransacIterationsForConfidence(0.005, 8), 10000);
    EXPECT_EQ(ransacIterationsForConfidence(1e-5, 4), 10000);
    EXPECT_EQ(ransacIterationsForConfidence(0.0, 4), 10000);
}

TEST(ransac_iterations, prosac_does_not_repeat_the_initial_sample)
{
    // GIVEN: a PROSAC pool containing exactly one sample's worth of points
    // WHEN/THEN: only one iteration is spent on it, since every draw would be identical
    EXPECT_EQ(prosacIterationsPerPoolSize(4, 4), 1);
    EXPECT_EQ(prosacIterationsPerPoolSize(8, 8), 1);

    // AND: a pool one larger has only sample_size distinct samples containing the newest point
    EXPECT_EQ(prosacIterationsPerPoolSize(5, 4), 4);

    // AND: larger pools are capped
    EXPECT_EQ(prosacIterationsPerPoolSize(50, 4), 10);
}

TEST(ransac_essential_matrix, ransac_compiles)
{
    std::vector<correspondence> matches;
    essential_matrix_model model;
    std::vector<bool> inliers;

    double score = ransac(matches, model, inliers);

    EXPECT_EQ(score, 0);
    EXPECT_EQ(inliers.size(), 0);
}

TEST(ransac_essential_matrix, fits_identity)
{
    std::vector<correspondence> matches;
    matches.push_back(correspondence{Eigen::Vector3d{1, 2, 1}, Eigen::Vector3d{1, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 2, 1}, Eigen::Vector3d{2, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 1, 1}, Eigen::Vector3d{2, 1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 1, 1}, Eigen::Vector3d{1, 1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 2, 3}, Eigen::Vector3d{1, 2, 3}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 2, 2}, Eigen::Vector3d{2, 2, 2}});
    matches.push_back(correspondence{Eigen::Vector3d{3, 1, 2}, Eigen::Vector3d{3, 1, 2}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 3, 2}, Eigen::Vector3d{1, 3, 2}});
    for (auto &m : matches)
    {
        m.measurement1.normalize();
        m.measurement2.normalize();
    }

    essential_matrix_model model;
    std::vector<bool> inliers;

    double score = ransac(matches, model, inliers);

    EXPECT_GE(score, 0.16);
    EXPECT_EQ(inliers.size(), 8);
    EXPECT_GE(std::count(inliers.begin(), inliers.end(), true), 1);
}

TEST(ransac_essential_matrix, fitInliers_uses_correct_subset)
{
    std::vector<correspondence> matches;
    matches.push_back(correspondence{Eigen::Vector3d{1, 2, 1}, Eigen::Vector3d{1, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{100, 200, 1}, Eigen::Vector3d{200, 100, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 2, 1}, Eigen::Vector3d{2, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{150, 250, 1}, Eigen::Vector3d{250, 150, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 1, 1}, Eigen::Vector3d{2, 1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 1, 1}, Eigen::Vector3d{1, 1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1.5, 1.5, 1}, Eigen::Vector3d{1.5, 1.5, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 2, 3}, Eigen::Vector3d{1, 2, 3}});
    matches.push_back(correspondence{Eigen::Vector3d{3, 1, 2}, Eigen::Vector3d{3, 1, 2}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 3, 2}, Eigen::Vector3d{1, 3, 2}});
    for (auto &m : matches)
    {
        m.measurement1.normalize();
        m.measurement2.normalize();
    }

    std::vector<bool> inliers = {true, false, true, false, true, true, true, true, true, true};

    essential_matrix_model model;
    model.fitInliers(matches, inliers);

    double inlier_error = 0;
    size_t inlier_count = 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (inliers[i])
        {
            inlier_error += std::abs(model.error(matches[i]));
            inlier_count++;
        }
    }
    double avg_inlier_error = inlier_error / inlier_count;

    double outlier_error = 0;
    size_t outlier_count = 0;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (!inliers[i])
        {
            outlier_error += std::abs(model.error(matches[i]));
            outlier_count++;
        }
    }
    double avg_outlier_error = outlier_error / outlier_count;

    EXPECT_LT(avg_inlier_error, 0.01);
    EXPECT_GT(avg_outlier_error, avg_inlier_error * 2);
}

TEST(ransac_homography, fitInliers_with_too_few_points_keeps_model)
{
    // GIVEN: a homography fitted from 4 points
    EssentialScene scene = makeEssentialScene(10);
    homography_model model;
    model.fit(scene.matches, {0, 1, 2, 3});
    const Eigen::Matrix3d fitted = model.homography;

    // WHEN: refitting from only 3 inliers
    std::vector<bool> inliers(scene.matches.size(), false);
    inliers[4] = inliers[5] = inliers[6] = true;
    model.fitInliers(scene.matches, inliers);

    // THEN: the underdetermined refit is skipped
    EXPECT_TRUE(model.homography.isApprox(fitted));
}

TEST(ransac_homography, fitInliers_is_least_squares_on_noisy_points)
{
    // GIVEN: noisy correspondences of a plane
    std::vector<Eigen::Vector3d> points;
    for (size_t i = 0; i < 60; i++)
        points.push_back(scenePoint(i, 0));
    EssentialScene scene = makeEssentialScene(points);
    std::mt19937 rng(3);
    std::normal_distribution<double> noise(0, 1e-3);
    for (auto &m : scene.matches)
        m.measurement2 = (m.measurement2.hnormalized() + Eigen::Vector2d(noise(rng), noise(rng))).homogeneous();

    // WHEN: fitting on all of them
    homography_model model;
    model.fitInliers(scene.matches, std::vector<bool>(scene.matches.size(), true));

    // THEN: the least-squares fit explains the data at least as well as the true plane homography
    const Eigen::Matrix3d R = Eigen::AngleAxisd(0.2, Eigen::Vector3d(0.3, -1, 0.5).normalized()).toRotationMatrix();
    homography_model truth;
    truth.homography = R + Eigen::Vector3d(1, 0.2, -0.3) * Eigen::Vector3d(0, 0, 1.0 / 8).transpose();
    truth.homography_inverse = truth.homography.inverse();
    auto rms = [&](homography_model &h) {
        double sum_sq = 0;
        for (const auto &m : scene.matches)
            sum_sq += std::pow(h.error(m), 2);
        return std::sqrt(sum_sq / scene.matches.size());
    };
    EXPECT_LE(rms(model), rms(truth));
}

TEST(ransac_fundamental_matrix, fitInliers_recovers_ground_truth)
{
    // GIVEN: noise-free correspondences from a known relative pose, in normalized coordinates so F == E
    EssentialScene scene = makeEssentialScene(20);
    std::vector<bool> inliers(scene.matches.size(), true);

    // WHEN: fitting on all of them
    fundamental_matrix_model model;
    model.fitInliers(scene.matches, inliers);

    // THEN: the ground truth is recovered and every match has ~zero error
    EXPECT_LT(essentialDistanceUpToScale(model.fundamental_matrix, scene.E), 1e-6);
    for (const auto &m : scene.matches)
        EXPECT_LT(model.error(m), 1e-9);
}

TEST(ransac_fundamental_matrix, checkDegeneracy_recovers_F_from_dominant_plane)
{
    // GIVEN: a scene dominated by a plane, with the off-plane points listed first
    std::vector<Eigen::Vector3d> points;
    for (size_t i = 0; i < 8; i++)
        points.push_back(scenePoint(i, 0) * 0.5);
    for (size_t i = 8; i < 48; i++)
    {
        Eigen::Vector3d p = scenePoint(i, 0);
        p.z() += 0.1 * p.x();
        points.push_back(p);
    }
    EssentialScene scene = makeEssentialScene(points);

    // AND: a model which currently explains none of it
    fundamental_matrix_model model;
    model.fundamental_matrix.setZero();
    std::vector<bool> inliers(scene.matches.size(), true);

    // WHEN: checking for plane degeneracy
    model.checkDegeneracy(scene.matches, inliers);

    // THEN: F is recovered from the plane homography and the off-plane epipole
    EXPECT_LT(essentialDistanceUpToScale(model.fundamental_matrix, scene.E), 1e-6);
    EXPECT_EQ(std::count(inliers.begin(), inliers.end(), true), 48);
}

TEST(ransac_essential_matrix, fitInliers_recovers_ground_truth)
{
    // GIVEN: noise-free correspondences from a known relative pose
    EssentialScene scene = makeEssentialScene(20);
    std::vector<bool> inliers(scene.matches.size(), true);

    // WHEN: fitting on all of them
    essential_matrix_model model;
    model.fitInliers(scene.matches, inliers);

    // THEN: the ground truth essential matrix is recovered and every match has ~zero error
    EXPECT_LT(essentialDistanceUpToScale(model.essential_matrix, scene.E), 1e-6);
    for (const auto &m : scene.matches)
        EXPECT_LT(model.error(m), 1e-9);
}

TEST(ransac_essential_matrix, ransac_recovers_ground_truth)
{
    // GIVEN: noise-free correspondences from a known relative pose
    EssentialScene scene = makeEssentialScene(30);

    // WHEN: running ransac
    essential_matrix_model model;
    std::vector<bool> inliers;
    ransac(scene.matches, model, inliers);

    // THEN: every match is an inlier of the ground truth essential matrix
    EXPECT_EQ(std::count(inliers.begin(), inliers.end(), true), 30);
    EXPECT_LT(essentialDistanceUpToScale(model.essential_matrix, scene.E), 1e-6);
}

INSTANTIATE_TEST_SUITE_P(
    ransac, ransac_p,
    ::testing::Combine(testing::Values(Eigen::Quaterniond::Identity(),
                                       Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitZ()))
                                       // TODO: get these working
                                       // Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX())),
                                       //  Eigen::Quaterniond(Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()))
                                       ),
                       testing::Values(Eigen::Vector3d::Zero(), Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(1, -1, 0),
                                       Eigen::Vector3d(-1, 1, 0), Eigen::Vector3d(-1, -1, 0))));
