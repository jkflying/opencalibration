#include <opencalibration/io/serialize.hpp>

#include <Eigen/Dense>
#include <gdal.h>
#include <gdal_priv.h>
#include <memory>
#include <spdlog/spdlog.h>
#include <type_traits>
#include <vector>

namespace
{
std::mutex m_registration;
struct gdal_registration
{
    gdal_registration()
    {
        static bool registered = false;
        if (!registered)
        {
            registered = true;
            std::lock_guard<std::mutex> lock(m_registration);
            GDALAllRegister();
        }
    }
};

class GDALDatasetDeleter
{
  public:
    void operator()(GDALDataset *dataset) const
    {
        if (dataset)
            GDALClose(dataset);
    }
};

class GDALRasterBandDeleter
{
  public:
    void operator()(GDALRasterBand *band) const
    {
        // Do any necessary cleanup here
        if (band)
            GDALClose(band);
    }
};

using GDALDatasetPtr = std::unique_ptr<GDALDataset, GDALDatasetDeleter>;
using GDALRasterBandPtr = std::unique_ptr<GDALRasterBand, GDALRasterBandDeleter>;

} // namespace

namespace opencalibration
{

template <typename PixelValue> GDALDataType getGDALDataType()
{
    if (std::is_same<PixelValue, uint8_t>::value)
    {
        return GDT_Byte;
    }
    else if (std::is_same<PixelValue, uint16_t>::value)
    {
        return GDT_UInt16;
    }
    else if (std::is_same<PixelValue, uint32_t>::value)
    {
        return GDT_UInt32;
    }
    else if (std::is_same<PixelValue, int8_t>::value)
    {
        return GDT_Byte;
    }
    else if (std::is_same<PixelValue, int16_t>::value)
    {
        return GDT_Int16;
    }
    else if (std::is_same<PixelValue, int32_t>::value)
    {
        return GDT_Int32;
    }
    else if (std::is_same<PixelValue, float>::value)
    {
        return GDT_Float32;
    }
    else if (std::is_same<PixelValue, double>::value)
    {
        return GDT_Float64;
    }
    else
    {
        // Unsupported pixel value type
        throw std::invalid_argument("Unsupported PixelValue type.");
    }
}

GDALColorInterp convertColorEnumToGDAL(Band color)
{
    switch (color)
    {
    case Band::GREY:
        return GCI_GrayIndex;
    case Band::RED:
        return GCI_RedBand;
    case Band::GREEN:
        return GCI_GreenBand;
    case Band::BLUE:
        return GCI_BlueBand;
    case Band::ALPHA:
        return GCI_AlphaBand;
    default:
        // Map all other colors to GCI_Undefined
        return GCI_Undefined;
    }
}

template <typename PixelValue>
std::string convertMultiLayerRasterToTIFF(const MultiLayerRaster<PixelValue> &multiLayerRaster)
{
    static gdal_registration registration;

    // order the bands into the expected order
    std::vector<std::pair<Band, size_t>> bandOrder(multiLayerRaster.layers.size());
    for (size_t i = 0; i < multiLayerRaster.layers.size(); i++)
    {
        bandOrder[i] = {multiLayerRaster.layers[i].band, i};
    }
    std::sort(bandOrder.begin(), bandOrder.end());

    // Create an in-memory TIFF dataset
    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (driver == nullptr)
    {
        spdlog::error("Error: GTiff driver not available.");
        return "";
    }
    const char *path = "/vsimem/mlr.tiff";
    GDALDatasetPtr dataset(driver->Create(path, multiLayerRaster.layers[0].pixels.cols(),
                                          multiLayerRaster.layers[0].pixels.rows(), multiLayerRaster.layers.size(),
                                          getGDALDataType<PixelValue>(), nullptr));

    if (!dataset)
    {
        spdlog::error("Error: Failed to create in-memory dataset.");
        return "";
    }

    // initialize the gdal dataset

    // copy the pixels into the gdal dataset
    // Copy the pixels into the GDAL dataset
    for (size_t i = 0; i < bandOrder.size(); ++i)
    {
        const auto &layer = multiLayerRaster.layers[bandOrder[i].second];
        GDALRasterBand *band = dataset->GetRasterBand(i + 1);
        if (!band)
        {
            spdlog::error("Error: Failed to get raster band.");
            return "";
        }
        PixelValue *data = const_cast<PixelValue *>(layer.pixels.data());
        const CPLErr err = band->RasterIO(GF_Write, 0, 0, layer.pixels.cols(), layer.pixels.rows(), data,
                                          layer.pixels.cols(), layer.pixels.rows(), GDT_Float32, 0, 0);
        band->SetColorInterpretation(convertColorEnumToGDAL(layer.band));
        if (err != CE_None)
        {
            spdlog::error("Error: failed to copy raster");
            return "";
        }
    }

    // Serialize the GDAL dataset into an in-memory string
    vsi_l_offset buffSize = 0;
    GByte *buf = VSIGetMemFileBuffer(path, &buffSize, false);

    std::string tiffString(reinterpret_cast<char *>(buf), static_cast<size_t>(buffSize));

    return tiffString;
}

std::string toTiff(const GenericRaster &raster)
{
    return std::visit([](const auto &multiLayerRaster) { return convertMultiLayerRasterToTIFF(multiLayerRaster); },
                      raster);
}

} // namespace opencalibration
