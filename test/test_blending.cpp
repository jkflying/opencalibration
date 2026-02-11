#include <opencalibration/ortho/blending.hpp>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

using namespace opencalibration::orthomosaic;

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
    EXPECT_FALSE(buf.at(1, 2, 3).valid);
}

TEST(Blending, no_ringing_at_shared_vertical_edge)
{
    // GIVEN: All layers share the same validity boundary (left 3/4 valid, right 1/4 invalid).
    // Invalid regions are black (0,0,0) in color, simulating the pipeline init.
    // Without pull-push fill this creates ringing artifacts from the Laplacian pyramid.
    int sz = 128;
    int num_layers = 3;
    float L_value = 50.0f;
    int boundary_col = 3 * sz / 4;

    std::vector<cv::Mat> lab_layers(num_layers);
    std::vector<cv::Mat> weight_maps(num_layers);

    for (int i = 0; i < num_layers; i++)
    {
        lab_layers[i] = cv::Mat(sz, sz, CV_32FC3, cv::Scalar(0, 0, 0));
        weight_maps[i] = cv::Mat::zeros(sz, sz, CV_32FC1);

        for (int r = 0; r < sz; r++)
        {
            for (int c = 0; c < boundary_col; c++)
            {
                lab_layers[i].at<cv::Vec3f>(r, c) = cv::Vec3f(L_value, 0.0f, 0.0f);
                weight_maps[i].at<float>(r, c) = 1.0f;
            }
        }
    }

    // WHEN: We Laplacian blend
    cv::Mat result = laplacianBlend(lab_layers, weight_maps, 4);
    ASSERT_FALSE(result.empty());

    // THEN: All valid pixels should have consistent color with no ringing
    int check_row = sz / 2;
    cv::Vec4b ref = result.at<cv::Vec4b>(check_row, sz / 4);

    for (int c = 0; c < boundary_col; c++)
    {
        cv::Vec4b pixel = result.at<cv::Vec4b>(check_row, c);
        for (int ch = 0; ch < 3; ch++)
        {
            EXPECT_NEAR(pixel[ch], ref[ch], 2) << "Ringing at col=" << c << " ch=" << ch;
        }
    }
}

TEST(Blending, no_ringing_at_shared_corner_edge)
{
    // GIVEN: All layers share the same invalid corner region (bottom-right).
    // This pattern matches the real-world case from blending_tile225.
    int sz = 128;
    int num_layers = 3;
    float L_value = 50.0f;

    std::vector<cv::Mat> lab_layers(num_layers);
    std::vector<cv::Mat> weight_maps(num_layers);

    for (int i = 0; i < num_layers; i++)
    {
        lab_layers[i] = cv::Mat(sz, sz, CV_32FC3, cv::Scalar(0, 0, 0));
        weight_maps[i] = cv::Mat::zeros(sz, sz, CV_32FC1);

        for (int r = 0; r < sz; r++)
        {
            for (int c = 0; c < sz; c++)
            {
                // Valid in the top-left 3/4-by-3/4 region.
                if (r < 3 * sz / 4 && c < 3 * sz / 4)
                {
                    lab_layers[i].at<cv::Vec3f>(r, c) = cv::Vec3f(L_value, 0.0f, 0.0f);
                    weight_maps[i].at<float>(r, c) = 1.0f;
                }
            }
        }
    }

    cv::Mat result = laplacianBlend(lab_layers, weight_maps, 4);
    ASSERT_FALSE(result.empty());

    // THEN: Pixels well inside the valid region should have uniform color
    cv::Vec4b ref = result.at<cv::Vec4b>(sz / 4, sz / 4);
    for (int r = 5; r < sz / 2; r++)
    {
        for (int c = 5; c < sz / 2; c++)
        {
            cv::Vec4b pixel = result.at<cv::Vec4b>(r, c);
            for (int ch = 0; ch < 3; ch++)
            {
                EXPECT_NEAR(pixel[ch], ref[ch], 2) << "Ringing at r=" << r << " c=" << c << " ch=" << ch;
            }
        }
    }
}

TEST(Blending, no_seam_at_layer_boundary)
{
    // GIVEN: Two layers with different colors and non-overlapping valid regions
    // that share a boundary. With per-layer fill, the extrapolated colors from each
    // layer leak through at coarser pyramid levels, creating a visible seam.
    // With consensus fill, both layers' invalid regions get the consensus color,
    // which eliminates the seam.
    int sz = 128;
    int boundary_col = sz / 2;

    cv::Mat lab_a(sz, sz, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat lab_b(sz, sz, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weight_a = cv::Mat::zeros(sz, sz, CV_32FC1);
    cv::Mat weight_b = cv::Mat::zeros(sz, sz, CV_32FC1);

    cv::Vec3f color_a(60.0f, 20.0f, 15.0f);
    cv::Vec3f color_b(40.0f, -15.0f, -10.0f);

    for (int r = 0; r < sz; r++)
    {
        for (int c = 0; c < sz; c++)
        {
            if (c < boundary_col)
            {
                lab_a.at<cv::Vec3f>(r, c) = color_a;
                weight_a.at<float>(r, c) = 1.0f;
            }
            else
            {
                lab_b.at<cv::Vec3f>(r, c) = color_b;
                weight_b.at<float>(r, c) = 1.0f;
            }
        }
    }

    cv::Mat result = laplacianBlend({lab_a, lab_b}, {weight_a, weight_b}, 4);
    ASSERT_FALSE(result.empty());

    // THEN: Far left should be pure layer A, far right should be pure layer B
    cv::Vec4b left_ref = result.at<cv::Vec4b>(sz / 2, 10);
    cv::Vec4b right_ref = result.at<cv::Vec4b>(sz / 2, sz - 11);

    for (int c = 5; c < sz / 4; c++)
    {
        cv::Vec4b pixel = result.at<cv::Vec4b>(sz / 2, c);
        for (int ch = 0; ch < 3; ch++)
        {
            EXPECT_NEAR(pixel[ch], left_ref[ch], 3)
                << "Seam artifact from layer B leaking into left at col=" << c << " ch=" << ch;
        }
    }

    for (int c = 3 * sz / 4; c < sz - 5; c++)
    {
        cv::Vec4b pixel = result.at<cv::Vec4b>(sz / 2, c);
        for (int ch = 0; ch < 3; ch++)
        {
            EXPECT_NEAR(pixel[ch], right_ref[ch], 3)
                << "Seam artifact from layer A leaking into right at col=" << c << " ch=" << ch;
        }
    }

    // Transition region (around boundary) should be monotonic - no ringing
    int check_row = sz / 2;
    for (int c = boundary_col - 20; c < boundary_col + 19; c++)
    {
        cv::Vec4b curr = result.at<cv::Vec4b>(check_row, c);
        cv::Vec4b next = result.at<cv::Vec4b>(check_row, c + 1);
        EXPECT_GE(curr[0], next[0] - 1) << "Non-monotonic transition at col=" << c << " (possible ringing)";
    }
}

TEST(Blending, boundary_only_secondary_blending)
{
    // GIVEN: Two layers where primary cameras differ on left vs right half.
    // A transition mask suppresses the secondary layer except near the boundary.
    int sz = 128;
    int boundary_col = sz / 2;
    int transition_radius = 16;

    cv::Vec3f color_a(60.0f, 10.0f, 5.0f);
    cv::Vec3f color_b(40.0f, -10.0f, -5.0f);

    // Layer 0 (primary): color_a on left, color_b on right — full weight everywhere
    cv::Mat lab_0(sz, sz, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weight_0 = cv::Mat::zeros(sz, sz, CV_32FC1);
    for (int r = 0; r < sz; r++)
    {
        for (int c = 0; c < sz; c++)
        {
            lab_0.at<cv::Vec3f>(r, c) = (c < boundary_col) ? color_a : color_b;
            weight_0.at<float>(r, c) = 1.0f;
        }
    }

    // Layer 1 (secondary): opposite colors — weight masked by boundary distance
    cv::Mat lab_1(sz, sz, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weight_1 = cv::Mat::zeros(sz, sz, CV_32FC1);
    for (int r = 0; r < sz; r++)
    {
        for (int c = 0; c < sz; c++)
        {
            lab_1.at<cv::Vec3f>(r, c) = (c < boundary_col) ? color_b : color_a;
            weight_1.at<float>(r, c) = 1.0f;
        }
    }

    // Build boundary mask from layer 0's camera assignment (left vs right)
    cv::Mat boundary_mask(sz, sz, CV_8UC1, cv::Scalar(0));
    for (int r = 0; r < sz; r++)
    {
        for (int c = 0; c < sz; c++)
        {
            int my_cam = (c < boundary_col) ? 0 : 1;
            const int dx[] = {-1, 1, 0, 0};
            const int dy[] = {0, 0, -1, 1};
            for (int d = 0; d < 4; d++)
            {
                int nr = r + dy[d], nc = c + dx[d];
                if (nr >= 0 && nr < sz && nc >= 0 && nc < sz)
                {
                    int neighbor_cam = (nc < boundary_col) ? 0 : 1;
                    if (neighbor_cam != my_cam)
                    {
                        boundary_mask.at<uint8_t>(r, c) = 255;
                        break;
                    }
                }
            }
        }
    }

    cv::Mat inv_boundary;
    cv::bitwise_not(boundary_mask, inv_boundary);
    cv::Mat boundary_dist;
    cv::distanceTransform(inv_boundary, boundary_dist, cv::DIST_L2, 3);

    float steepness = std::log(99.0f) / static_cast<float>(transition_radius);
    for (int r = 0; r < sz; r++)
    {
        for (int c = 0; c < sz; c++)
        {
            float d = boundary_dist.at<float>(r, c);
            float falloff = 1.0f / (1.0f + std::exp(steepness * d));
            weight_1.at<float>(r, c) *= falloff;
        }
    }

    // WHEN: Laplacian blend
    cv::Mat result = laplacianBlend({lab_0, lab_1}, {weight_0, weight_1}, 4);
    ASSERT_FALSE(result.empty());

    // THEN: Interior pixels (far from boundary) should match primary layer exactly
    // because secondary weight is 0 there.
    cv::Vec4b ref_left = result.at<cv::Vec4b>(sz / 2, 10);
    cv::Vec4b ref_right = result.at<cv::Vec4b>(sz / 2, sz - 11);

    // Check well inside the left half — use generous margin from transition zone
    // because the Laplacian pyramid spreads some influence beyond weight boundaries
    int margin = transition_radius + 16;
    for (int c = 5; c < boundary_col - margin; c++)
    {
        cv::Vec4b pixel = result.at<cv::Vec4b>(sz / 2, c);
        for (int ch = 0; ch < 3; ch++)
        {
            EXPECT_NEAR(pixel[ch], ref_left[ch], 2) << "Interior left mismatch at col=" << c << " ch=" << ch;
        }
    }

    // Check well inside the right half
    for (int c = boundary_col + margin; c < sz - 5; c++)
    {
        cv::Vec4b pixel = result.at<cv::Vec4b>(sz / 2, c);
        for (int ch = 0; ch < 3; ch++)
        {
            EXPECT_NEAR(pixel[ch], ref_right[ch], 2) << "Interior right mismatch at col=" << c << " ch=" << ch;
        }
    }

    // Check that the boundary region has a smooth transition (monotonic L channel)
    for (int c = boundary_col - transition_radius; c < boundary_col + transition_radius - 1; c++)
    {
        cv::Vec4b curr = result.at<cv::Vec4b>(sz / 2, c);
        cv::Vec4b next = result.at<cv::Vec4b>(sz / 2, c + 1);
        // color_a is brighter (L=60) on left, color_b darker (L=40) on right
        // So pixel brightness should decrease or stay the same going left to right
        EXPECT_GE(curr[0], next[0] - 1) << "Non-monotonic transition at col=" << c;
    }
}
