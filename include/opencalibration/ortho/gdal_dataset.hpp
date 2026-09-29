#pragma once

#include <gdal.h>

#include <memory>
#include <string>

namespace opencalibration::orthomosaic
{

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

using GDALDatasetPtr = std::unique_ptr<void, GDALDatasetDeleter>;

inline GDALDatasetPtr openGDALDataset(const std::string &path)
{
    GDALAllRegister();
    GDALDatasetH dataset = GDALOpen(path.c_str(), GA_ReadOnly);
    return GDALDatasetPtr(dataset);
}

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

    GDALRasterBandH GetRasterBand(int nBand)
    {
        return GDALGetRasterBand(m_handle, nBand);
    }

  private:
    GDALDatasetH m_handle;
};

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

    void GetBlockSize(int *pnXSize, int *pnYSize) const
    {
        GDALGetBlockSize(m_handle, pnXSize, pnYSize);
    }

  private:
    GDALRasterBandH m_handle;
};

} // namespace opencalibration::orthomosaic
