#include <opencalibration/distort/distort_keypoints.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

TEST(distort, image_to_3d_to_image)
{
    CameraModel model;
    model.focal_length_pixels = 6000;
    model.pixels_cols = 4000;
    model.pixels_rows = 3000;
    model.principle_point = Eigen::Vector2d(model.pixels_cols, model.pixels_rows) / 2;

    for (size_t i = 0; i < model.pixels_cols; i+=10)
    {
        for (size_t j = 0; j < model.pixels_rows; j+=10)
        {
            Eigen::Vector2d p(i,j);
            Eigen::Vector2d p2 = image_from_3d(image_to_3d(p, model), model);

            EXPECT_NEAR(p.x(), p2.x(), 1e-9);
            EXPECT_NEAR(p.y(), p2.y(), 1e-9);
        }
    }
}
