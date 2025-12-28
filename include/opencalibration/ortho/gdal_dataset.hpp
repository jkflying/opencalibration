#pragma once

#include <gdal/gdal_priv.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace opencalibration::orthomosaic
{

// Custom deleter for GDALDataset
struct GDALDatasetDeleter
{
    void operator()(GDALDataset *dataset) const
    {
        if (dataset)
        {
            GDALClose(dataset);
        }
    }
};

// RAII wrapper for GDALDataset using unique_ptr
using GDALDatasetPtr = std::unique_ptr<GDALDataset, GDALDatasetDeleter>;

// Helper function to open a GDAL dataset for reading
inline GDALDatasetPtr openGDALDataset(const std::string &path)
{
    GDALAllRegister();
    GDALDataset *dataset = static_cast<GDALDataset *>(GDALOpen(path.c_str(), GA_ReadOnly));
    return GDALDatasetPtr(dataset);
}

} // namespace opencalibration::orthomosaic
