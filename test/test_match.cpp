#include <opencalibration/extract/extract_features.hpp>
#include <opencalibration/match/match_features.hpp>

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

using namespace opencalibration;

TEST(match, finds_correspondences)
{
    // GIVEN: two sets of match_features
    std::string path1 = TEST_DATA_DIR "P2530253.JPG";
    std::string path2 = TEST_DATA_DIR "P2540254.JPG";

    std::vector<feature_2d> feat1 = opencalibration::extract_features(path1);
    std::vector<feature_2d> feat2 = opencalibration::extract_features(path2);

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
            const feature_2d& f1 = feat1[m.feature_index_1];
            const feature_2d& f2 = feat2[m.feature_index_2];

            cv::circle(img_combo, toPoint(f1.location), 30, cv::Scalar(0, 0, 255), 5);
            cv::circle(img_combo, toPoint(f2.location + img2_offset), 30, cv::Scalar(0, 0, 255), 5);
            cv::line(img_combo, toPoint(f1.location), toPoint(f2.location + img2_offset), cv::Scalar(0, 0, 255), 5);
        }
        std::string output_path = TEST_DATA_OUTPUT_DIR "matches_overlay.jpg";
        cv::imwrite(output_path, img_combo);
    }
}
