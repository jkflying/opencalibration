#pragma once

#include <gdal.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace opencalibration::orthomosaic
{

// Custom deleter for GDALDataset
struct GDALDatasetDeleter
{
    void operator()(GDALDatasetH dataset) const
    {
        if (dataset)
        {
            GDALClose(dataset);
        }
    }
};

// RAII wrapper for GDALDataset using unique_ptr
// Note: GDALDatasetH is void*, so we use void as the managed type
using GDALDatasetPtr = std::unique_ptr<void, GDALDatasetDeleter>;

// Helper function to open a GDAL dataset for reading
inline GDALDatasetPtr openGDALDataset(const std::string &path)
{
    GDALAllRegister();
    GDALDatasetH dataset = GDALOpen(path.c_str(), GA_ReadOnly);
    return GDALDatasetPtr(dataset);
}

// Wrapper class to provide convenient access to GDALDatasetH
class GDALDatasetWrapper
{
  public:
    explicit GDALDatasetWrapper(GDALDatasetH handle) : m_handle(handle)
    {
    }

    [[nodiscard]] int GetRasterXSize() const
    {
        return GDALGetRasterXSize(m_handle);
    }
    [[nodiscard]] int GetRasterYSize() const
    {
        return GDALGetRasterYSize(m_handle);
    }
    [[nodiscard]] int GetRasterCount() const
    {
        return GDALGetRasterCount(m_handle);
    }

    [[nodiscard]] const char *GetProjectionRef() const
    {
        return GDALGetProjectionRef(m_handle);
    }

    CPLErr GetGeoTransform(double *padfTransform) const
    {
        return GDALGetGeoTransform(m_handle, padfTransform);
    }

    CPLErr SetGeoTransform(double *padfTransform)
    {
        return GDALSetGeoTransform(m_handle, padfTransform);
    }

    CPLErr SetProjection(const char *pszProjection)
    {
        return GDALSetProjection(m_handle, pszProjection);
    }

    GDALRasterBandH GetRasterBand(int nBand)
    {
        return GDALGetRasterBand(m_handle, nBand);
    }

    CPLErr BuildOverviews(const char *pszResampling, int nOverviews, const int *panOverviewList,
                          GDALProgressFunc pfnProgress = nullptr, void *pProgressData = nullptr)
    {
        return GDALBuildOverviews(m_handle, pszResampling, nOverviews, panOverviewList, 0, nullptr, pfnProgress,
                                  pProgressData);
    }

    [[nodiscard]] GDALDatasetH Handle() const
    {
        return m_handle;
    }

  private:
    GDALDatasetH m_handle;
};

// Wrapper class to provide convenient access to GDALRasterBandH
class GDALRasterBandWrapper
{
  public:
    explicit GDALRasterBandWrapper(GDALRasterBandH handle) : m_handle(handle)
    {
    }

    [[nodiscard]] GDALColorInterp GetColorInterpretation() const
    {
        return GDALGetRasterColorInterpretation(m_handle);
    }

    CPLErr SetColorInterpretation(GDALColorInterp eInterp)
    {
        return GDALSetRasterColorInterpretation(m_handle, eInterp);
    }

    CPLErr RasterIO(GDALRWFlag eRWFlag, int nXOff, int nYOff, int nXSize, int nYSize, void *pData, int nBufXSize,
                    int nBufYSize, GDALDataType eBufType, int nPixelSpace, int nLineSpace)
    {
        return GDALRasterIO(m_handle, eRWFlag, nXOff, nYOff, nXSize, nYSize, pData, nBufXSize, nBufYSize, eBufType,
                            nPixelSpace, nLineSpace);
    }

    CPLErr SetNoDataValue(double dfValue)
    {
        // Note: GDALSetRasterNoDataValue returns an error, use GDALSetRasterNoDataValue for C API
        GDALSetRasterNoDataValue(m_handle, dfValue);
        return CE_None; // Assume success for now
    }

    void GetBlockSize(int *pnXSize, int *pnYSize) const
    {
        GDALGetBlockSize(m_handle, pnXSize, pnYSize);
    }

    [[nodiscard]] GDALRasterBandH Handle() const
    {
        return m_handle;
    }

  private:
    GDALRasterBandH m_handle;
};

} // namespace opencalibration::orthomosaic
