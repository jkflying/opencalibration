#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/extract/extract_features.hpp>
#include <opencalibration/match/match_features.hpp>
#include <opencalibration/model_inliers/ransac.hpp>

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

using namespace opencalibration;

TEST(ransac, homography_filter_correspondences)
{
    // GIVEN: matches between two images and some dumb camera models
    std::string path1 = TEST_DATA_DIR "P2540254.JPG";
    std::string path2 = TEST_DATA_DIR "P2530253.JPG";

    std::vector<feature_2d> feat1 = opencalibration::extract_features(path1);
    std::vector<feature_2d> feat2 = opencalibration::extract_features(path2);
    std::vector<feature_match> matches = match_features(feat1, feat2);
    CameraModel cam_model;
    cam_model.pixels_rows = 4016;
    cam_model.pixels_cols = 5344;
    cam_model.focal_length_pixels = 5000;
    cam_model.principle_point = Eigen::Vector2d(cam_model.pixels_cols, cam_model.pixels_rows) * 0.5;

    std::vector<correspondence> correspondences = distort_keypoints(feat1, feat2, matches, cam_model, cam_model);

    // WHEN: we filter them with a homography
    homography_model model;
    std::vector<bool> inliers;
    double score = ransac(correspondences, model, inliers);

    // THEN: there should be roughly the right number
    EXPECT_GT(score, 0.5);
    std::cout << "s: " << score << " m: " << std::endl << model.homography << std::endl;

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

        for (size_t i = 0; i < matches.size(); i++) //const feature_match &m : matches)
        {
            const feature_match& m = matches[i];
            const feature_2d &f1 = feat1[m.feature_index_1];
            const feature_2d &f2 = feat2[m.feature_index_2];

            cv::Scalar color = inliers[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            int width = inliers[i] ? 5 : 1;
            cv::circle(img_combo, toPoint(f1.location), 30, color, width);
            cv::circle(img_combo, toPoint(f2.location + img2_offset), 30, color, width);
            cv::line(img_combo, toPoint(f1.location), toPoint(f2.location + img2_offset), color, width);
        }
        std::string output_path = TEST_DATA_OUTPUT_DIR "inlier_matches_overlay.jpg";
        cv::imwrite(output_path, img_combo);
    }
}
