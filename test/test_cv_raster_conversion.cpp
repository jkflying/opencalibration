#include <opencalibration/io/cv_raster_conversion.hpp>

#include <opencv2/core.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

TEST(cv_raster_conversion, cvToRaster_uint8)
{
    cv::Mat mat(2, 3, CV_8UC1, cv::Scalar(42));
    GenericRaster raster = cvToRaster(mat);
    auto *r = std::get_if<MultiLayerRaster<uint8_t>>(&raster);
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(1u, r->layers.size());
    EXPECT_EQ(42, r->layers[0].pixels(0, 0));
}

TEST(cv_raster_conversion, cvToRaster_int8)
{
    cv::Mat mat(2, 3, CV_8SC1, cv::Scalar(-10));
    GenericRaster raster = cvToRaster(mat);
    auto *r = std::get_if<MultiLayerRaster<int8_t>>(&raster);
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(-10, r->layers[0].pixels(0, 0));
}

TEST(cv_raster_conversion, cvToRaster_uint16)
{
    cv::Mat mat(2, 3, CV_16UC1, cv::Scalar(1000));
    GenericRaster raster = cvToRaster(mat);
    auto *r = std::get_if<MultiLayerRaster<uint16_t>>(&raster);
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(1000, r->layers[0].pixels(0, 0));
}

TEST(cv_raster_conversion, cvToRaster_int16)
{
    cv::Mat mat(2, 3, CV_16SC1, cv::Scalar(-500));
    GenericRaster raster = cvToRaster(mat);
    auto *r = std::get_if<MultiLayerRaster<int16_t>>(&raster);
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(-500, r->layers[0].pixels(0, 0));
}

TEST(cv_raster_conversion, cvToRaster_int32)
{
    cv::Mat mat(2, 3, CV_32SC1, cv::Scalar(100000));
    GenericRaster raster = cvToRaster(mat);
    auto *r = std::get_if<MultiLayerRaster<int32_t>>(&raster);
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(100000, r->layers[0].pixels(0, 0));
}

TEST(cv_raster_conversion, cvToRaster_float)
{
    cv::Mat mat(2, 3, CV_32FC1, cv::Scalar(3.14f));
    GenericRaster raster = cvToRaster(mat);
    auto *r = std::get_if<MultiLayerRaster<float>>(&raster);
    ASSERT_NE(nullptr, r);
    EXPECT_FLOAT_EQ(3.14f, r->layers[0].pixels(0, 0));
}

TEST(cv_raster_conversion, cvToRaster_double_converts_to_float)
{
    cv::Mat mat(2, 3, CV_64FC1, cv::Scalar(2.718));
    GenericRaster raster = cvToRaster(mat);
    auto *r = std::get_if<MultiLayerRaster<float>>(&raster);
    ASSERT_NE(nullptr, r);
    EXPECT_NEAR(2.718f, r->layers[0].pixels(0, 0), 1e-5);
}

TEST(cv_raster_conversion, rasterToCv_uint8_multilayer)
{
    MultiLayerRaster<uint8_t> raster(2, 3, 1);
    raster.layers[0].pixels(0, 0) = 99;
    raster.layers[0].pixels(1, 2) = 200;

    GenericRaster generic = raster;
    cv::Mat mat = rasterToCv(generic);

    EXPECT_EQ(CV_8U, mat.depth());
    EXPECT_EQ(1, mat.channels());
    EXPECT_EQ(99, mat.at<uint8_t>(0, 0));
    EXPECT_EQ(200, mat.at<uint8_t>(1, 2));
}

TEST(cv_raster_conversion, rasterToCv_float_multilayer)
{
    MultiLayerRaster<float> raster(2, 3, 1);
    raster.layers[0].pixels(0, 0) = 1.5f;

    GenericRaster generic = raster;
    cv::Mat mat = rasterToCv(generic);

    EXPECT_EQ(CV_32F, mat.depth());
    EXPECT_FLOAT_EQ(1.5f, mat.at<float>(0, 0));
}

TEST(cv_raster_conversion, rasterToCv_layer)
{
    RasterLayer<int16_t> layer;
    layer.band = Band::GREY;
    layer.pixels.resize(2, 3);
    layer.pixels(0, 0) = -42;

    GenericLayer generic = layer;
    cv::Mat mat = rasterToCv(generic);

    EXPECT_EQ(CV_16S, mat.depth());
    EXPECT_EQ(-42, mat.at<int16_t>(0, 0));
}

TEST(cv_raster_conversion, rgb_channel_reorder_roundtrip)
{
    MultiLayerRaster<uint8_t> raster(2, 2, 3);
    raster.layers[0].band = Band::RED;
    raster.layers[1].band = Band::GREEN;
    raster.layers[2].band = Band::BLUE;
    raster.layers[0].pixels.setConstant(10);
    raster.layers[1].pixels.setConstant(20);
    raster.layers[2].pixels.setConstant(30);

    GenericRaster generic = raster;
    cv::Mat mat = rasterToCv(generic);
    EXPECT_EQ(3, mat.channels());

    // OpenCV BGR order: channel 0=B(30), 1=G(20), 2=R(10)
    cv::Vec3b pixel = mat.at<cv::Vec3b>(0, 0);
    EXPECT_EQ(30, pixel[0]);
    EXPECT_EQ(20, pixel[1]);
    EXPECT_EQ(10, pixel[2]);
}

TEST(cv_raster_conversion, RasterToRGB_uint8)
{
    MultiLayerRaster<uint8_t> raster(2, 2, 3);
    raster.layers[0].band = Band::RED;
    raster.layers[1].band = Band::GREEN;
    raster.layers[2].band = Band::BLUE;
    raster.layers[0].pixels.setConstant(100);
    raster.layers[1].pixels.setConstant(150);
    raster.layers[2].pixels.setConstant(200);

    GenericRaster generic = raster;
    RGBRaster rgb = RasterToRGB(generic);

    EXPECT_EQ(3u, rgb.layers.size());
    EXPECT_EQ(Band::RED, rgb.layers[0].band);
    EXPECT_EQ(100, rgb.layers[0].pixels(0, 0));
    EXPECT_EQ(150, rgb.layers[1].pixels(0, 0));
    EXPECT_EQ(200, rgb.layers[2].pixels(0, 0));
}

TEST(cv_raster_conversion, RasterToRGB_int16_casts)
{
    MultiLayerRaster<int16_t> raster(2, 2, 3);
    raster.layers[0].band = Band::RED;
    raster.layers[1].band = Band::GREEN;
    raster.layers[2].band = Band::BLUE;
    raster.layers[0].pixels.setConstant(100);
    raster.layers[1].pixels.setConstant(150);
    raster.layers[2].pixels.setConstant(200);

    GenericRaster generic = raster;
    RGBRaster rgb = RasterToRGB(generic);

    EXPECT_EQ(100, rgb.layers[0].pixels(0, 0));
    EXPECT_EQ(150, rgb.layers[1].pixels(0, 0));
    EXPECT_EQ(200, rgb.layers[2].pixels(0, 0));
}

TEST(cv_raster_conversion, cvToRaster_3channel_bgr_labels)
{
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));
    GenericRaster raster = cvToRaster(mat);
    auto *r = std::get_if<MultiLayerRaster<uint8_t>>(&raster);
    ASSERT_NE(nullptr, r);
    ASSERT_EQ(3u, r->layers.size());
    EXPECT_EQ(Band::BLUE, r->layers[0].band);
    EXPECT_EQ(Band::GREEN, r->layers[1].band);
    EXPECT_EQ(Band::RED, r->layers[2].band);
}
