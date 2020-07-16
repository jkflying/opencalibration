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
}
