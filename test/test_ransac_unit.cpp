#include <opencalibration/model_inliers/ransac.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

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
