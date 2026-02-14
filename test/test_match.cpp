#include <opencalibration/extract/extract_features.hpp>
#include <opencalibration/match/match_features.hpp>

#include <gtest/gtest.h>

#include <eigen3/Eigen/Geometry>
#include <opencv2/opencv.hpp>

using namespace opencalibration;

TEST(match, finds_correspondences)
{
    // GIVEN: two sets of match_features
    std::string path1 = TEST_DATA_DIR "P2540254.JPG";
    std::string path2 = TEST_DATA_DIR "P2530253.JPG";

    std::vector<feature_2d> feat1 = opencalibration::extract_features(cv::imread(path1));
    std::vector<feature_2d> feat2 = opencalibration::extract_features(cv::imread(path2));

    // WHEN: we find the matches
    std::vector<feature_match> matches = match_features(feat1, feat2);

    // THEN: there should be roughly the right number
    EXPECT_GT(matches.size(), 50);

    // AND: they should correspond (visual inspection)
    bool visual_inspection = false;
#if defined(VISUAL_INSPECTION)
    visual_inspection = true;
#endif
    if (visual_inspection)
    {
        cv::Mat img1 = cv::imread(path1);
        cv::Mat img2 = cv::imread(path2);

        cv::Mat img_combo;
        cv::hconcat(img1, img2, img_combo);

        Eigen::Vector2d img2_offset(img1.size().width, 0);

        auto toPoint = [](const Eigen::Vector2d &p) -> cv::Point2d {
            cv::Point2d cvp;
            cvp.x = p.x();
            cvp.y = p.y();
            return cvp;
        };
        for (const feature_match &m : matches)
        {
            const feature_2d &f1 = feat1[m.feature_index_1];
            const feature_2d &f2 = feat2[m.feature_index_2];

            cv::circle(img_combo, toPoint(f1.location), 30, cv::Scalar(0, 0, 255), 5);
            cv::circle(img_combo, toPoint(f2.location + img2_offset), 30, cv::Scalar(0, 0, 255), 5);
            cv::line(img_combo, toPoint(f1.location), toPoint(f2.location + img2_offset), cv::Scalar(0, 0, 255), 5);
        }
        std::string output_path = TEST_DATA_OUTPUT_DIR "matches_overlay.jpg";
        cv::imwrite(output_path, img_combo);
    }
}

TEST(match, subset_matches_use_original_indices)
{
    std::string path1 = TEST_DATA_DIR "P2540254.JPG";
    std::string path2 = TEST_DATA_DIR "P2530253.JPG";

    std::vector<feature_2d> feat1 = extract_features(cv::imread(path1));
    std::vector<feature_2d> feat2 = extract_features(cv::imread(path2));

    std::vector<size_t> indices1 = spatially_subsample_feature_indices(feat1, 40.0);
    std::vector<size_t> indices2 = spatially_subsample_feature_indices(feat2, 40.0);

    ASSERT_GT(indices1.size(), 10);
    ASSERT_GT(indices2.size(), 10);

    std::vector<feature_match> subset_matches = match_features_subset(feat1, feat2, indices1, indices2);

    EXPECT_GT(subset_matches.size(), 5);

    for (const auto &m : subset_matches)
    {
        EXPECT_LT(m.feature_index_1, feat1.size());
        EXPECT_LT(m.feature_index_2, feat2.size());

        bool index1_in_subset = std::find(indices1.begin(), indices1.end(), m.feature_index_1) != indices1.end();
        bool index2_in_subset = std::find(indices2.begin(), indices2.end(), m.feature_index_2) != indices2.end();

        EXPECT_TRUE(index1_in_subset) << "Match index1=" << m.feature_index_1 << " not in subset indices";
        EXPECT_TRUE(index2_in_subset) << "Match index2=" << m.feature_index_2 << " not in subset indices";
    }
}

TEST(match, local_guided_respects_homography)
{
    std::string path1 = TEST_DATA_DIR "P2540254.JPG";
    std::string path2 = TEST_DATA_DIR "P2530253.JPG";

    std::vector<feature_2d> feat1 = extract_features(cv::imread(path1));
    std::vector<feature_2d> feat2 = extract_features(cv::imread(path2));

    std::vector<feature_match> full_matches = match_features(feat1, feat2);
    ASSERT_GT(full_matches.size(), 50);

    Eigen::Matrix3d identity_homography = Eigen::Matrix3d::Identity();
    const double search_radius = 50.0;

    std::vector<feature_match> guided_matches =
        match_features_local_guided(feat1, feat2, identity_homography, search_radius);

    EXPECT_GT(guided_matches.size(), 10);

    for (const auto &m : guided_matches)
    {
        Eigen::Vector2d predicted = (identity_homography * feat1[m.feature_index_1].location.homogeneous()).hnormalized();
        Eigen::Vector2d actual = feat2[m.feature_index_2].location;
        double distance = (predicted - actual).norm();

        EXPECT_LT(distance, search_radius) << "Match at distance " << distance << " exceeds search radius "
                                            << search_radius;
    }
}

TEST(match, local_guided_with_translation)
{
    std::string path1 = TEST_DATA_DIR "P2540254.JPG";
    std::string path2 = TEST_DATA_DIR "P2530253.JPG";

    std::vector<feature_2d> feat1 = extract_features(cv::imread(path1));
    std::vector<feature_2d> feat2 = extract_features(cv::imread(path2));

    Eigen::Matrix3d translation_homography = Eigen::Matrix3d::Identity();
    translation_homography(0, 2) = 100.0;
    translation_homography(1, 2) = 50.0;

    const double search_radius = 30.0;

    std::vector<feature_match> guided_matches =
        match_features_local_guided(feat1, feat2, translation_homography, search_radius);

    for (const auto &m : guided_matches)
    {
        Eigen::Vector2d predicted = (translation_homography * feat1[m.feature_index_1].location.homogeneous()).hnormalized();
        Eigen::Vector2d actual = feat2[m.feature_index_2].location;
        double distance = (predicted - actual).norm();

        EXPECT_LT(distance, search_radius);
    }
}

TEST(match, spatial_subsample_reduces_feature_count)
{
    std::string path1 = TEST_DATA_DIR "P2540254.JPG";
    std::vector<feature_2d> features = extract_features(cv::imread(path1));

    ASSERT_GT(features.size(), 100);

    std::vector<size_t> subsampled_40px = spatially_subsample_feature_indices(features, 40.0);
    std::vector<size_t> subsampled_80px = spatially_subsample_feature_indices(features, 80.0);
    std::vector<size_t> subsampled_160px = spatially_subsample_feature_indices(features, 160.0);

    EXPECT_LT(subsampled_40px.size(), features.size());
    EXPECT_LT(subsampled_80px.size(), subsampled_40px.size());
    EXPECT_LT(subsampled_160px.size(), subsampled_80px.size());

    for (size_t idx : subsampled_160px)
    {
        EXPECT_LT(idx, features.size());
    }
}

TEST(match, spatial_subsample_maintains_spacing)
{
    std::string path1 = TEST_DATA_DIR "P2540254.JPG";
    std::vector<feature_2d> features = extract_features(cv::imread(path1));

    const double spacing = 60.0;
    std::vector<size_t> subsampled = spatially_subsample_feature_indices(features, spacing);

    ASSERT_GT(subsampled.size(), 5);

    for (size_t i = 0; i < subsampled.size(); i++)
    {
        for (size_t j = i + 1; j < subsampled.size(); j++)
        {
            double distance = (features[subsampled[i]].location - features[subsampled[j]].location).norm();
            EXPECT_GT(distance, spacing) << "Features at indices " << i << " and " << j << " are closer than spacing";
        }
    }
}

TEST(match, spatial_subsample_keeps_strongest_features)
{
    std::vector<feature_2d> features;

    features.push_back(feature_2d{{100, 100}, 0.5f, {}});
    features.push_back(feature_2d{{110, 105}, 0.9f, {}});
    features.push_back(feature_2d{{200, 200}, 0.3f, {}});
    features.push_back(feature_2d{{205, 202}, 0.7f, {}});
    features.push_back(feature_2d{{300, 300}, 0.4f, {}});

    const double spacing = 20.0;
    std::vector<size_t> subsampled = spatially_subsample_feature_indices(features, spacing);

    ASSERT_EQ(subsampled.size(), 3);

    EXPECT_EQ(subsampled[0], 1);
    EXPECT_EQ(subsampled[1], 3);
    EXPECT_EQ(subsampled[2], 4);

    for (size_t kept_idx : subsampled)
    {
        const auto &kept_feature = features[kept_idx];

        for (size_t other_idx = 0; other_idx < features.size(); other_idx++)
        {
            if (other_idx == kept_idx)
                continue;

            const auto &other_feature = features[other_idx];
            double distance = (kept_feature.location - other_feature.location).norm();

            if (distance <= spacing)
            {
                EXPECT_GE(kept_feature.strength, other_feature.strength)
                    << "Kept feature at " << kept_idx << " (strength=" << kept_feature.strength
                    << ") should have >= strength than nearby feature at " << other_idx
                    << " (strength=" << other_feature.strength << ") at distance " << distance;
            }
        }
    }
}
