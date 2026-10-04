#include <opencalibration/extract/extract_features.hpp>

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

using namespace opencalibration;
TEST(extract_features, gives_points)
{
    // GIVEN: a path
    std::string path = TEST_DATA_DIR "IMG_1378_RGB.jpg";

    // WHEN: we extract the features
    auto extracted = opencalibration::extract_features(cv::imread(path));
    const auto &res = extracted.features;

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

TEST(extract_features, dense_features_are_nms_rejected)
{
    // GIVEN: an image
    std::string path = TEST_DATA_DIR "IMG_1378_RGB.jpg";
    auto extracted = opencalibration::extract_features(cv::imread(path));

    // THEN: there should be dense features after the sparse ones
    EXPECT_GT(extracted.features.size(), extracted.num_sparse_features);

    // AND: total features should be more than sparse alone
    EXPECT_GT(extracted.features.size(), extracted.num_sparse_features);

    // AND: dense features should each be within NMS radius of some kept feature
    for (size_t i = extracted.num_sparse_features; i < extracted.features.size(); i++)
    {
        const auto &df = extracted.features[i];
        double min_dist = std::numeric_limits<double>::max();
        for (size_t j = 0; j < extracted.num_sparse_features; j++)
        {
            double dist = (df.location - extracted.features[j].location).norm();
            min_dist = std::min(min_dist, dist);
        }
        EXPECT_LT(min_dist, 200) << "Dense feature too far from any kept feature";
    }
}

TEST(extract_features, empty_image_returns_empty)
{
    auto extracted = opencalibration::extract_features(cv::Mat());
    EXPECT_EQ(extracted.features.size(), 0);
    EXPECT_EQ(extracted.num_sparse_features, 0);
}

TEST(extract_features, downscaled_locations_have_no_half_pixel_bias)
{
    // GIVEN: a large image with a single symmetric blob at a known pixel centre
    const cv::Point2d centre(2000, 1500);
    cv::Mat image(3000, 4000, CV_8UC3, cv::Scalar(40, 40, 40));
    cv::circle(image, cv::Point(int(centre.x), int(centre.y)), 40, cv::Scalar(220, 220, 220), cv::FILLED, cv::LINE_AA);
    cv::GaussianBlur(image, image, cv::Size(0, 0), 8);

    // WHEN: we extract features, which downsamples the image internally
    auto extracted = opencalibration::extract_features(image);
    ASSERT_FALSE(extracted.features.empty());

    // THEN: the feature nearest the blob should be at its centre in full resolution pixel coordinates
    const Eigen::Vector2d expected(centre.x, centre.y);
    double min_dist = std::numeric_limits<double>::max();
    for (const auto &f : extracted.features)
    {
        min_dist = std::min(min_dist, (f.location - expected).norm());
    }
    EXPECT_LT(min_dist, 0.5);
}

TEST(extract_features, strongest_feature_is_only_sparse)
{
    // GIVEN: an image
    std::string path = TEST_DATA_DIR "IMG_1378_RGB.jpg";

    // WHEN: we extract the features
    auto extracted = opencalibration::extract_features(cv::imread(path));
    ASSERT_GT(extracted.num_sparse_features, 0);

    // THEN: the strongest feature should not also appear among the dense features
    const auto &strongest = extracted.features[0];
    for (size_t i = extracted.num_sparse_features; i < extracted.features.size(); i++)
    {
        EXPECT_NE(extracted.features[i].location, strongest.location) << "Dense feature " << i;
    }
}
