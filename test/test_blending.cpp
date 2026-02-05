#include <opencalibration/ortho/blending.hpp>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <cmath>

using namespace opencalibration::orthomosaic;

TEST(Blending, outlier_rejection_no_outlier)
{
    // GIVEN: 3 layers with consistent colors
    LayeredTileBuffer tile;
    tile.resize(2, 2, 3);

    for (int layer = 0; layer < 3; layer++)
    {
        for (int i = 0; i < 4; i++)
        {
            tile.layers[layer][i].color_bgr = cv::Vec3b(100, 100, 100);
            tile.layers[layer][i].valid = true;
        }
    }

    // WHEN: We run outlier rejection
    rejectOutliers(tile, 3.0);

    // THEN: All samples should still be valid
    for (int layer = 0; layer < 3; layer++)
    {
        for (int i = 0; i < 4; i++)
        {
            EXPECT_TRUE(tile.layers[layer][i].valid);
        }
    }
}

TEST(Blending, outlier_rejection_with_outlier)
{
    // GIVEN: 3 layers where one layer has a very different brightness
    LayeredTileBuffer tile;
    tile.resize(1, 1, 3);

    // Layer 0 and 1: similar dark color
    tile.at(0, 0, 0).color_bgr = cv::Vec3b(50, 50, 50);
    tile.at(0, 0, 0).valid = true;
    tile.at(1, 0, 0).color_bgr = cv::Vec3b(55, 55, 55);
    tile.at(1, 0, 0).valid = true;
    // Layer 2: very bright outlier
    tile.at(2, 0, 0).color_bgr = cv::Vec3b(250, 250, 250);
    tile.at(2, 0, 0).valid = true;

    // WHEN: We run outlier rejection
    rejectOutliers(tile, 3.0);

    // THEN: Layer 2 should be rejected as outlier
    EXPECT_TRUE(tile.at(0, 0, 0).valid);
    EXPECT_TRUE(tile.at(1, 0, 0).valid);
    EXPECT_FALSE(tile.at(2, 0, 0).valid);
}

TEST(Blending, outlier_rejection_needs_3_layers)
{
    // GIVEN: Only 2 valid layers
    LayeredTileBuffer tile;
    tile.resize(1, 1, 3);

    tile.at(0, 0, 0).color_bgr = cv::Vec3b(50, 50, 50);
    tile.at(0, 0, 0).valid = true;
    tile.at(1, 0, 0).color_bgr = cv::Vec3b(250, 250, 250);
    tile.at(1, 0, 0).valid = true;
    tile.at(2, 0, 0).valid = false; // Invalid

    // WHEN: We run outlier rejection
    rejectOutliers(tile, 3.0);

    // THEN: Both valid layers should remain (can't reject with only 2)
    EXPECT_TRUE(tile.at(0, 0, 0).valid);
    EXPECT_TRUE(tile.at(1, 0, 0).valid);
}

TEST(Blending, compute_blend_weight_center)
{
    // GIVEN: A pixel at the center of a 100x100 image
    float weight = computeBlendWeight(50, 50, 100, 100, 10.0f);

    // THEN: Weight should be positive and relatively high
    EXPECT_GT(weight, 0.0f);
}

TEST(Blending, compute_blend_weight_edge)
{
    // GIVEN: A pixel at the edge of a 100x100 image
    float weight_edge = computeBlendWeight(0, 50, 100, 100, 10.0f);
    float weight_center = computeBlendWeight(50, 50, 100, 100, 10.0f);

    // THEN: Edge weight should be less than center weight
    EXPECT_LT(weight_edge, weight_center);
}

TEST(Blending, compute_blend_weight_proximity)
{
    // GIVEN: Same pixel position but different camera distances
    float weight_near = computeBlendWeight(50, 50, 100, 100, 5.0f);
    float weight_far = computeBlendWeight(50, 50, 100, 100, 50.0f);

    // THEN: Closer camera should have higher weight
    EXPECT_GT(weight_near, weight_far);
}

TEST(Blending, laplacian_blend_single_layer)
{
    // GIVEN: A single solid-color layer
    cv::Mat lab_layer(64, 64, CV_32FC3, cv::Scalar(128, 128, 128));
    cv::Mat weight(64, 64, CV_32FC1, cv::Scalar(1.0));

    // WHEN: We blend a single layer
    cv::Mat result = laplacianBlend({lab_layer}, {weight}, 3);

    // THEN: Result should be non-empty BGRA
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.channels(), 4);
    EXPECT_EQ(result.rows, 64);
    EXPECT_EQ(result.cols, 64);
}

TEST(Blending, laplacian_blend_two_layers_smooth)
{
    // GIVEN: Two layers with different colors, split weights
    int sz = 64;
    cv::Mat lab_a(sz, sz, CV_32FC3, cv::Scalar(180, 128, 128)); // Bright
    cv::Mat lab_b(sz, sz, CV_32FC3, cv::Scalar(80, 128, 128));  // Dark
    cv::Mat weight_a = cv::Mat::zeros(sz, sz, CV_32FC1);
    cv::Mat weight_b = cv::Mat::zeros(sz, sz, CV_32FC1);

    // Left half: layer A, right half: layer B
    for (int r = 0; r < sz; r++)
    {
        for (int c = 0; c < sz; c++)
        {
            if (c < sz / 2)
            {
                weight_a.at<float>(r, c) = 1.0f;
            }
            else
            {
                weight_b.at<float>(r, c) = 1.0f;
            }
        }
    }

    // WHEN: We Laplacian blend
    cv::Mat result = laplacianBlend({lab_a, lab_b}, {weight_a, weight_b}, 4);

    // THEN: Result should exist and have smooth transition
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.channels(), 4);

    // Check that the left side is different from the right side
    cv::Vec4b left_pixel = result.at<cv::Vec4b>(sz / 2, sz / 4);
    cv::Vec4b right_pixel = result.at<cv::Vec4b>(sz / 2, 3 * sz / 4);

    // The left should be brighter than the right (at least in some channel)
    // Since LAB 180 is brighter than 80
    int left_intensity = left_pixel[0] + left_pixel[1] + left_pixel[2];
    int right_intensity = right_pixel[0] + right_pixel[1] + right_pixel[2];
    EXPECT_NE(left_intensity, right_intensity);
}

TEST(Blending, laplacian_blend_empty)
{
    // GIVEN: Empty layers
    cv::Mat result = laplacianBlend({}, {}, 3);

    // THEN: Result should be empty
    EXPECT_TRUE(result.empty());
}

TEST(Blending, layered_tile_buffer_resize)
{
    LayeredTileBuffer buf;
    buf.resize(10, 20, 3);

    EXPECT_EQ(buf.width, 10);
    EXPECT_EQ(buf.height, 20);
    EXPECT_EQ(buf.num_layers, 3);
    EXPECT_EQ(buf.layers.size(), 3u);
    EXPECT_EQ(buf.layers[0].size(), 200u);

    // Default samples should be invalid
    EXPECT_FALSE(buf.at(0, 0, 0).valid);
    EXPECT_FALSE(buf.at(2, 19, 9).valid);
}

TEST(Blending, layered_tile_buffer_at)
{
    LayeredTileBuffer buf;
    buf.resize(5, 5, 2);

    buf.at(0, 2, 3).color_bgr = cv::Vec3b(10, 20, 30);
    buf.at(0, 2, 3).valid = true;

    EXPECT_EQ(buf.at(0, 2, 3).color_bgr, cv::Vec3b(10, 20, 30));
    EXPECT_TRUE(buf.at(0, 2, 3).valid);
    EXPECT_FALSE(buf.at(1, 2, 3).valid); // Different layer
}
