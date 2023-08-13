#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/distort/invert_distortion.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

template <typename Model1, typename Model2>
void verifyIsSame(const Model1 &model1, const Model2 &model2, double eps = 1e-12)
{
    size_t num_failed = 0;
    for (size_t i = 0; i < model1.pixels_cols; i += model1.pixels_cols / 20)
    {
        for (size_t j = 0; j < model1.pixels_rows; j += model1.pixels_rows / 20)
        {
            Eigen::Vector2d p(i, j);
            Eigen::Vector2d p2 = image_from_3d(image_to_3d(p, model1), model2);

            EXPECT_NEAR(p.x(), p2.x(), eps);
            EXPECT_NEAR(p.y(), p2.y(), eps);

            if (std::abs(p.x() - p2.x()) > eps || std::abs(p.y() - p2.y()) > eps)
                num_failed++;
        }
    }
    EXPECT_EQ(num_failed, 0);
}

template <typename Model> void verify(const Model &model, double eps = 1e-12)
{
    EXPECT_NO_FATAL_FAILURE(verifyIsSame(model, model, eps));
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
    model.radial_distortion << 0.02, -0.07, 0.1;

    EXPECT_NO_FATAL_FAILURE(verify(model, 1e-2)); // only solve to 1/100 of a pixel error
}

TEST(distort, image_to_3d_to_image_radial_tangential_distortion)
{
    CameraModel model;
    model.focal_length_pixels = 6000;
    model.pixels_cols = 4000;
    model.pixels_rows = 3000;
    model.principle_point = Eigen::Vector2d(model.pixels_cols, model.pixels_rows) / 2;
    model.radial_distortion << 0.02, -0.07, 0.1;
    model.tangential_distortion << 0.08, -0.08;

    EXPECT_NO_FATAL_FAILURE(verify(model, 1e-2)); // only solve to 1/100 of a pixel error
}

TEST(distort, inversemodel_image_to_3d_to_image_no_distortion)
{
    InverseCameraModel model;
    model.focal_length_pixels = 6000;
    model.pixels_cols = 4000;
    model.pixels_rows = 3000;
    model.principle_point = Eigen::Vector2d(model.pixels_cols, model.pixels_rows) / 2;

    EXPECT_NO_FATAL_FAILURE(verify(model, 1e-12)); // simple model inverts perfectly
}

TEST(distort, inversemodel_image_to_3d_to_image_radial_tangential_distortion)
{
    InverseCameraModel model;
    model.focal_length_pixels = 6000;
    model.pixels_cols = 4000;
    model.pixels_rows = 3000;
    model.principle_point = Eigen::Vector2d(model.pixels_cols, model.pixels_rows) / 2;
    model.radial_distortion << 0.02, -0.07, 0.1;
    model.tangential_distortion << 0.08, -0.08;

    EXPECT_NO_FATAL_FAILURE(verify(model, 1e-2)); // only solve to 1/100 of a pixel error
}

TEST(distort, inversemodel_conversion_to_model_no_distortion)
{
    DifferentiableCameraModel<double> model;
    model.focal_length_pixels = 6000;
    model.pixels_cols = 4000;
    model.pixels_rows = 3000;
    model.principle_point = Eigen::Vector2d(model.pixels_cols, model.pixels_rows) / 2;

    const InverseDifferentiableCameraModel<double> inverse_model = convertModel(model);

    EXPECT_NO_FATAL_FAILURE(verifyIsSame(model, inverse_model, 1e-12)); // simple model inverts perfectly
    EXPECT_NO_FATAL_FAILURE(verifyIsSame(inverse_model, model, 1e-12)); // simple model inverts perfectly
}

TEST(distort, DISABLED_inversemodel_conversion_to_model_radial_tangential_distortion)
{
    DifferentiableCameraModel<double> model;
    model.focal_length_pixels = 6000;
    model.pixels_cols = 4000;
    model.pixels_rows = 3000;
    model.principle_point = Eigen::Vector2d(model.pixels_cols, model.pixels_rows) / 2;
    model.radial_distortion << 0.02, -0.07, 0.1;
    model.tangential_distortion << 0.08, -0.08;

    const InverseDifferentiableCameraModel<double> inverse_model = convertModel(model);

    EXPECT_NO_FATAL_FAILURE(verifyIsSame(model, inverse_model, 1e-2)); // only solve to 1/100 of a pixel error
    EXPECT_NO_FATAL_FAILURE(verifyIsSame(inverse_model, model, 1e-2)); // only solve to 1/100 of a pixel error
}
