#include <opencalibration/types/raster.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

TEST(raster, cast_uint8_to_float)
{
    MultiLayerRaster<uint8_t> src(2, 3, 2);
    src.layers[0].band = Band::RED;
    src.layers[1].band = Band::GREEN;
    src.layers[0].pixels(0, 0) = 255;
    src.layers[0].pixels(1, 2) = 128;
    src.layers[1].pixels(0, 1) = 42;

    auto dst = src.cast<float>();

    ASSERT_EQ(2u, dst.layers.size());
    EXPECT_EQ(Band::RED, dst.layers[0].band);
    EXPECT_EQ(Band::GREEN, dst.layers[1].band);
    EXPECT_FLOAT_EQ(255.f, dst.layers[0].pixels(0, 0));
    EXPECT_FLOAT_EQ(128.f, dst.layers[0].pixels(1, 2));
    EXPECT_FLOAT_EQ(42.f, dst.layers[1].pixels(0, 1));
    EXPECT_EQ(2, dst.layers[0].pixels.rows());
    EXPECT_EQ(3, dst.layers[0].pixels.cols());
}

TEST(raster, get_and_set)
{
    MultiLayerRaster<int16_t> raster(3, 4, 3);
    Eigen::Vector<int16_t, Eigen::Dynamic> pixel(3);
    pixel << 10, 20, 30;

    EXPECT_TRUE(raster.set(1, 2, pixel));
    EXPECT_EQ(10, raster.layers[0].pixels(1, 2));
    EXPECT_EQ(20, raster.layers[1].pixels(1, 2));
    EXPECT_EQ(30, raster.layers[2].pixels(1, 2));

    Eigen::Vector<int16_t, Eigen::Dynamic> out(3);
    EXPECT_TRUE(raster.get(1, 2, out));
    EXPECT_EQ(pixel, out);
}

TEST(raster, get_wrong_size_returns_false)
{
    MultiLayerRaster<uint8_t> raster(2, 2, 3);
    Eigen::Vector<uint8_t, Eigen::Dynamic> pixel(2);
    EXPECT_FALSE(raster.get(0, 0, pixel));
}

TEST(raster, set_wrong_size_returns_false)
{
    MultiLayerRaster<uint8_t> raster(2, 2, 3);
    Eigen::Vector<uint8_t, Eigen::Dynamic> pixel(4);
    EXPECT_FALSE(raster.set(0, 0, pixel));
}

TEST(raster, operator_eq_same_type)
{
    MultiLayerRaster<uint8_t> a(2, 2, 1);
    MultiLayerRaster<uint8_t> b(2, 2, 1);
    a.layers[0].pixels.setZero();
    b.layers[0].pixels.setZero();
    EXPECT_TRUE(a == b);

    b.layers[0].pixels(0, 0) = 1;
    EXPECT_FALSE(a == b);
}

TEST(raster, operator_eq_different_band_count)
{
    MultiLayerRaster<uint8_t> a(2, 2, 1);
    MultiLayerRaster<uint8_t> b(2, 2, 2);
    EXPECT_FALSE(a == b);
}

TEST(raster, operator_eq_different_size)
{
    MultiLayerRaster<uint8_t> a(2, 2, 1);
    MultiLayerRaster<uint8_t> b(3, 2, 1);
    EXPECT_FALSE(a == b);
}

TEST(raster, operator_eq_different_band_type)
{
    MultiLayerRaster<uint8_t> a(2, 2, 1);
    MultiLayerRaster<uint8_t> b(2, 2, 1);
    a.layers[0].band = Band::RED;
    b.layers[0].band = Band::GREEN;
    EXPECT_FALSE(a == b);
}

TEST(raster, operator_eq_different_pixel_types)
{
    MultiLayerRaster<uint8_t> a(2, 2, 1);
    MultiLayerRaster<float> b(2, 2, 1);
    EXPECT_FALSE(a == b);
}
