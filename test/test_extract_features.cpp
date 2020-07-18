#include <opencalibration/extract/extract_features.hpp>

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

using namespace opencalibration;
TEST(extract_features, gives_points)
{
    // GIVEN: a path
    std::string path = TEST_DATA_DIR "P2530253.JPG";

    // WHEN: we extract the features
    std::vector<feature_2d> res = opencalibration::extract_features(path);

    // THEN: there should be at least 100
    EXPECT_GT(res.size(), 100);

    // AND: they should be overlaid in nice locations (visual inspection)
    bool visual_inspection = false;
#if defined(VISUAL_INSPECTION)
    visual_inspection = true;
#endif
    if (visual_inspection)
    {
        cv::Mat img = cv::imread(path);

        auto toPoint = [](const Eigen::Vector2d &p) -> cv::Point2d {
            cv::Point2d cvp;
            cvp.x = p.x();
            cvp.y = p.y();
            return cvp;
        };
        for (const feature_2d &f : res)
        {
            cv::circle(img, toPoint(f.location), 30, cv::Scalar(0, 0, 255), 5);
        }
        std::string output_path = TEST_DATA_OUTPUT_DIR "features_overlay.jpg";
        cv::imwrite(output_path, img);
    }
}
