#include <opencalibration/model_inliers/ransac.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

TEST(ransac, homography_ransac_compiles)
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

TEST(ransac, homography_fits_identity)
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
    ASSERT_TRUE(model.decompose(matches, inliers, orientation, translation, orientation2, translation2));
    EXPECT_NEAR(translation.norm(), 0, 1e-14);
    EXPECT_NEAR(Eigen::AngleAxisd(orientation).angle(), 0, 1e-14);
}

TEST(ransac, homography_translation)
{
    // GIVEN: 4 correspondences, from square A to square B when A + (1,1,0) == B
    std::vector<correspondence> matches;
    matches.push_back(correspondence{Eigen::Vector3d{1, 2, 1}, Eigen::Vector3d{2, 3, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 2, 1}, Eigen::Vector3d{3, 3, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{2, 1, 1}, Eigen::Vector3d{3, 2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, 1, 1}, Eigen::Vector3d{2, 2, 1}});

    homography_model model;
    std::vector<bool> inliers;

    // WHEN: we get the ransac model
    double score = ransac(matches, model, inliers);

    // THEN: it should have a 1 score because we only have 100% inliers
    EXPECT_DOUBLE_EQ(score, 1);

    // AND: the model should be correct, and all the points inliers, and the model an identity
    EXPECT_EQ(inliers.size(), 4);
    EXPECT_EQ(std::count(inliers.begin(), inliers.end(), true), 4);
    Eigen::Matrix3d expected_homography;
    // clang-format off
    expected_homography << 1, 0, 1,
                           0, 1, 1,
                           0, 0, 1;
    // clang-format on
    EXPECT_NEAR((model.homography - expected_homography).norm(), 0, 1e-14) << model.homography;

    // AND: the decomposition should be an identity
    Eigen::Vector3d translation, translation2;
    Eigen::Quaterniond orientation, orientation2;
    ASSERT_TRUE(model.decompose(matches, inliers, orientation, translation, orientation2, translation2));
    EXPECT_NEAR((translation - Eigen::Vector3d(1, 1, 0)).norm(), 0, 1e-14) << translation.transpose() << std::endl
                                                                           << translation2.transpose();
    EXPECT_NEAR(Eigen::AngleAxisd(orientation).angle(), 0, 1e-14) << orientation.coeffs().transpose() << std::endl
                                                                  << orientation2.coeffs().transpose();
}

TEST(ransac, homography_rotation_z)
{
    // GIVEN: 4 correspondences, from square A to square B when A + (1,1,0) == B
    std::vector<correspondence> matches;
    matches.push_back(correspondence{Eigen::Vector3d{1, 1, 1}, Eigen::Vector3d{1, -1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, -1, 1}, Eigen::Vector3d{-1, -1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{-1, -1, 1}, Eigen::Vector3d{-1, 1, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{-1, 1, 1}, Eigen::Vector3d{1, 1, 1}});

    homography_model model;
    std::vector<bool> inliers;

    // WHEN: we get the ransac model
    double score = ransac(matches, model, inliers);

    // THEN: it should have a 1 score because we only have 100% inliers
    EXPECT_DOUBLE_EQ(score, 1);

    // AND: the model should be correct, and all the points inliers, and the model an identity
    EXPECT_EQ(inliers.size(), 4);
    EXPECT_EQ(std::count(inliers.begin(), inliers.end(), true), 4);
    Eigen::Matrix3d expected_homography;
    // clang-format off
    expected_homography << 0, 1, 0,
                          -1, 0, 0,
                           0, 0, 1;
    // clang-format on
    EXPECT_NEAR((model.homography - expected_homography).norm(), 0, 1e-14) << model.homography;

    // AND: the decomposition should be an identity
    Eigen::Vector3d translation, translation2;
    Eigen::Quaterniond orientation, orientation2;
    ASSERT_TRUE(model.decompose(matches, inliers, orientation, translation, orientation2, translation2));
    EXPECT_NEAR(translation.norm(), 0, 1e-14) << translation.transpose();
    EXPECT_NEAR(Eigen::AngleAxisd(orientation).angle(), M_PI_2, 1e-14) << orientation.coeffs().transpose();
    EXPECT_NEAR((Eigen::AngleAxisd(orientation).axis() + Eigen::Vector3d::UnitZ()).norm(), 0, 1e-14)
        << Eigen::AngleAxisd(orientation).axis();
}

TEST(ransac, homography_rotation_translation)
{
    // GIVEN: 4 correspondences, from square A to square B when A + (1,1,0) == B
    std::vector<correspondence> matches;
    matches.push_back(correspondence{Eigen::Vector3d{1, 1, 1}, Eigen::Vector3d{2, -2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{1, -1, 1}, Eigen::Vector3d{0, -2, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{-1, -1, 1}, Eigen::Vector3d{0, 0, 1}});
    matches.push_back(correspondence{Eigen::Vector3d{-1, 1, 1}, Eigen::Vector3d{2, 0, 1}});

    homography_model model;
    std::vector<bool> inliers;

    // WHEN: we get the ransac model
    double score = ransac(matches, model, inliers);

    // THEN: it should have a 1 score because we only have 100% inliers
    EXPECT_DOUBLE_EQ(score, 1);

    // AND: the model should be correct, and all the points inliers, and the model an identity
    EXPECT_EQ(inliers.size(), 4);
    EXPECT_EQ(std::count(inliers.begin(), inliers.end(), true), 4);
    Eigen::Matrix3d expected_homography;
    // clang-format off
    expected_homography << 0, 1, 1,
                          -1, 0,-1,
                           0, 0, 1;
    // clang-format on
    EXPECT_NEAR((model.homography - expected_homography).norm(), 0, 1e-14) << model.homography;

    // AND: the decomposition should be an identity
    Eigen::Vector3d translation, translation2;
    Eigen::Quaterniond orientation, orientation2;
    ASSERT_TRUE(model.decompose(matches, inliers, orientation, translation, orientation2, translation2));
    EXPECT_NEAR((translation - Eigen::Vector3d(1, -1, 0)).norm(), 0, 1e-14) << translation.transpose();
    EXPECT_NEAR(Eigen::AngleAxisd(orientation).angle(), M_PI_2, 1e-14) << orientation.coeffs().transpose();
    EXPECT_NEAR((Eigen::AngleAxisd(orientation).axis() + Eigen::Vector3d::UnitZ()).norm(), 0, 1e-14)
        << Eigen::AngleAxisd(orientation).axis();
}
