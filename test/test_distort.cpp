#include <opencalibration/distort/distort_keypoints.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

void verify(const CameraModel &model, double eps = 1e-12)
{
    size_t num_failed = 0;
    for (size_t i = 0; i < model.pixels_cols; i += 20)
    {
        for (size_t j = 0; j < model.pixels_rows; j += 20)
        {
            Eigen::Vector2d p(i, j);
            Eigen::Vector2d p2 = image_from_3d(image_to_3d(p, model), model);

            EXPECT_NEAR(p.x(), p2.x(), eps);
            EXPECT_NEAR(p.y(), p2.y(), eps);

            if (std::abs(p.x() - p2.x()) > eps || std::abs(p.y() - p2.y()) > eps)
                num_failed++;
        }
    }
    EXPECT_EQ(num_failed, 0);
}

TEST(distort, image_to_3d_to_image_no_distortion)
{
    CameraModel model;
    model.focal_length_pixels = 6000;
    model.pixels_cols = 4000;
    model.pixels_rows = 3000;
    model.principle_point = Eigen::Vector2d(model.pixels_cols, model.pixels_rows) / 2;

    EXPECT_NO_FATAL_FAILURE(verify(model, 1e-12)); // simple model inverts perfectly
}

TEST(distort, image_to_3d_to_image_radial_distortion)
{
    CameraModel model;
    model.focal_length_pixels = 6000;
    model.pixels_cols = 4000;
    model.pixels_rows = 3000;
    model.principle_point = Eigen::Vector2d(model.pixels_cols, model.pixels_rows) / 2;
    model.radial_distortion << 0, 0.02, 0, -0.07, 0, 0.1;

    EXPECT_NO_FATAL_FAILURE(verify(model, 1e-2)); // only solve to 1/100 of a pixel error
}

TEST(distort, image_to_3d_to_image_radial_tangential_distortion)
{
    CameraModel model;
    model.focal_length_pixels = 6000;
    model.pixels_cols = 4000;
    model.pixels_rows = 3000;
    model.principle_point = Eigen::Vector2d(model.pixels_cols, model.pixels_rows) / 2;
    model.radial_distortion << 0, 0.02, 0, -0.07, 0, 0.1;
    model.tangential_distortion << 0.08, -0.08;

    EXPECT_NO_FATAL_FAILURE(verify(model, 1e-2)); // only solve to 1/100 of a pixel error
}
