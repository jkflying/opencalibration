#include <opencalibration/io/cv_raster_conversion.hpp>
#include <opencv2/core.hpp>

using namespace opencalibration;
namespace
{
template <typename rasterType> int rasterTypeToCvType();

template <> int rasterTypeToCvType<uint8_t>()
{
    return CV_8U;
}

template <> int rasterTypeToCvType<int8_t>()
{
    return CV_8S;
}

template <> int rasterTypeToCvType<uint16_t>()
{
    return CV_16U;
}

template <> int rasterTypeToCvType<int16_t>()
{
    return CV_16S;
}

template <> int rasterTypeToCvType<int32_t>()
{
    return CV_32S;
}

template <> int rasterTypeToCvType<uint32_t>()
{
    return CV_32S;
}

template <> int rasterTypeToCvType<float>()
{
    return CV_32F;
}

template <typename T> cv::Mat rasterToCvImpl(const RasterLayer<T> &layer)
{
    const int cvType = CV_MAKETYPE(rasterTypeToCvType<T>(), 1);
    cv::Size dims(layer.pixels.rows(), layer.pixels.cols());
    cv::Mat result(dims, cvType);
    for (Eigen::Index row = 0; row < layer.pixels.rows(); row++)
    {
        for (Eigen::Index col = 0; col < layer.pixels.cols(); col++)
        {
            result.at<T>(row, col) = layer.pixels(row, col);
        }
    }

    return result;
}

template <typename T> cv::Mat rasterToCvImpl(const MultiLayerRaster<T> &raster)
{
    const int cvType = rasterTypeToCvType<T>();

    cv::Size dims(raster.layers[0].pixels.cols(), raster.layers[0].pixels.rows());

    std::vector<cv::Mat> channels;
    for (size_t i = 0; i < raster.layers.size(); i++)
    {
        channels.emplace_back(dims, cvType);
    }

    std::vector<size_t> channel_map;
    if (raster.layers.size() == 3 && raster.layers[0].band == Band::RED && raster.layers[1].band == Band::GREEN &&
        raster.layers[2].band == Band::BLUE)
    {
        channel_map = {2, 1, 0};
    }
    else
    {
        for (size_t i = 0; i < raster.layers.size(); i++)
        {
            channel_map.push_back(i);
        }
    }

    for (size_t channel = 0; channel < channel_map.size(); channel++)
    {
        for (Eigen::Index row = 0; row < raster.layers[channel].pixels.rows(); row++)
        {
            for (Eigen::Index col = 0; col < raster.layers[channel].pixels.cols(); col++)
            {
                // TODO: this could be better optimized
                channels[channel_map[channel]].at<T>(row, col) = raster.layers[channel].pixels(row, col);
            }
        }
    }

    cv::Mat result;
    cv::merge(channels, result);

    return result;
}

template <typename T> MultiLayerRaster<T> cvToRaster(const cv::Mat &mat)
{
    const cv::Size size = mat.size();
    MultiLayerRaster<T> raster(size.height, size.width, mat.channels());

    if (mat.channels() == 3 && mat.type() == CV_8UC3)
    {
        raster.layers[0].band = Band::BLUE;
        raster.layers[1].band = Band::GREEN;
        raster.layers[2].band = Band::RED;
    }

    std::vector<cv::Mat> channels;
    cv::split(mat, channels);

    for (size_t channel = 0; channel < raster.layers.size(); channel++)
    {
        for (Eigen::Index row = 0; row < raster.layers[channel].pixels.rows(); row++)
        {
            for (Eigen::Index col = 0; col < raster.layers[channel].pixels.cols(); col++)
            {
                raster.layers[channel].pixels(row, col) = channels[channel].at<T>(row, col);
            }
        }
    }
    return raster;
}

template <typename T> RGBRaster RasterToRGBImpl(const MultiLayerRaster<T> &raster)
{
    RGBRaster rgb(raster.layers[0].pixels.rows(), raster.layers[0].pixels.cols(), 3);
    rgb.layers[0].band = Band::RED;
    rgb.layers[1].band = Band::GREEN;
    rgb.layers[2].band = Band::BLUE;
    for (size_t i = 0; i < 3; i++)
    {
        size_t j = 0;
        for (; j < raster.layers.size(); j++)
        {
            if (raster.layers[j].band == rgb.layers[i].band)
            {
                break;
            }
        }
        // TODO: if T is uint8_t we can std::move the layers across instead
        rgb.layers[i].pixels = raster.layers[j].pixels.template cast<uint8_t>();
    }

    return rgb;
}
} // namespace

namespace opencalibration
{
cv::Mat rasterToCv(const GenericRaster &raster)
{
    auto visitor = [](const auto &unwrapped) -> cv::Mat { return rasterToCvImpl(unwrapped); };
    return std::visit(visitor, raster);
}

cv::Mat rasterToCv(const GenericLayer &layer)
{
    auto visitor = [](const auto &unwrapped) -> cv::Mat { return rasterToCvImpl(unwrapped); };
    return std::visit(visitor, layer);
}

GenericRaster cvToRaster(const cv::Mat &mat)
{
    switch (mat.depth())
    {
    case CV_8S:
        return ::cvToRaster<int8_t>(mat);
    case CV_8U:
        return ::cvToRaster<uint8_t>(mat);
    case CV_16U:
        return ::cvToRaster<uint16_t>(mat);
    case CV_16S:
        return ::cvToRaster<int16_t>(mat);
    case CV_32S:
        return ::cvToRaster<uint32_t>(mat);
    case CV_32F:
        return ::cvToRaster<float>(mat);
    case CV_64F:
        return ::cvToRaster<double>(mat).cast<float>();
    }

    throw std::runtime_error("Unsupported mat depth");
}

RGBRaster RasterToRGB(const GenericRaster &raster)
{
    auto visitor = [](const auto &unwrapped) -> RGBRaster { return RasterToRGBImpl(unwrapped); };
    return std::visit(visitor, raster);
}

} // namespace opencalibration
