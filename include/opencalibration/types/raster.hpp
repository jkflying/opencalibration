#pragma once

#include <Eigen/Core>

#include <variant>
#include <vector>

namespace opencalibration
{
enum class Band : uint8_t
{
    GREY = 0,
    RED,
    GREEN,
    BLUE,
    THERMAL,
    NIR,
    RED_EDGE,
    UV,
    ALPHA,
    COUNT,
    CAMERA_UUID
};

template <typename pixelValue> struct RasterLayer
{
    using PixelValue = pixelValue;

    Band band = Band::GREY;
    Eigen::Matrix<PixelValue, Eigen::Dynamic, Eigen::Dynamic> pixels;
};

template <typename pixelValue> struct MultiLayerRaster
{
    MultiLayerRaster() = default;
    MultiLayerRaster(int x, int y, int bands) : layers(bands)
    {
        for (int i = 0; i < bands; i++)
        {
            layers[i].pixels.resize(x, y);
        }
    }

    template <typename T> MultiLayerRaster<T> cast()
    {
        MultiLayerRaster<T> casted(layers[0].pixels.rows(), layers[0].pixels.cols(), layers.size());

        for (size_t layer = 0; layer < layers.size(); layer++)
        {
            casted.layers[layer].band = layers[layer].band;
            casted.layers[layer].pixels = layers[layer].pixels.template cast<T>();
        }

        return casted;
    }

    bool get(int row, int col, Eigen::Vector<pixelValue, Eigen::Dynamic> &pixel) const
    {
        if (pixel.size() != (Eigen::Index)layers.size())
            return false;

        for (Eigen::Index i = 0; i < pixel.size(); i++)
        {
            pixel(i) = layers[i].pixels(row, col);
        }
        return true;
    }

    bool set(int row, int col, const Eigen::Vector<pixelValue, Eigen::Dynamic> &pixel)
    {
        if (pixel.size() != (Eigen::Index)layers.size())
            return false;

        for (Eigen::Index i = 0; i < pixel.size(); i++)
        {
            layers[i].pixels(row, col) = pixel(i);
        }
        return true;
    }

    std::vector<RasterLayer<pixelValue>> layers;
};

template <typename pv1, typename pv2> bool operator==(const MultiLayerRaster<pv1> &a, const MultiLayerRaster<pv2> &b)
{
    if (!std::is_same<pv1, pv2>::value)
    {
        return false;
    }

    if (a.layers.size() != b.layers.size())
    {
        return false;
    }

    for (size_t i = 0; i < a.layers.size(); i++)
    {
        if (a.layers[i].band != b.layers[i].band)
        {
            return false;
        }

        if (a.layers[i].pixels.rows() != b.layers[i].pixels.rows() ||
            a.layers[i].pixels.cols() != b.layers[i].pixels.cols())
        {
            return false;
        }

        for (Eigen::Index j = 0; j < a.layers[i].pixels.size(); j++)
        {
            if (a.layers[i].pixels(j) != b.layers[i].pixels(j))
            {
                return false;
            }
        }
    }

    return true;
}

using RGBRaster = MultiLayerRaster<uint8_t>;
using GenericRaster = std::variant<MultiLayerRaster<uint8_t>, MultiLayerRaster<int8_t>, MultiLayerRaster<int16_t>,
                                   MultiLayerRaster<uint16_t>, MultiLayerRaster<int32_t>, MultiLayerRaster<float>>;
using GenericLayer = std::variant<RasterLayer<uint8_t>, RasterLayer<int8_t>, RasterLayer<uint16_t>,
                                  RasterLayer<int16_t>, RasterLayer<int32_t>, RasterLayer<float>>;
} // namespace opencalibration
