#pragma once

#include <opencalibration/surface/intersect.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/raster.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <jk/KDTree.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace opencalibration
{
class GeoCoord;
}

namespace opencalibration::orthomosaic
{
struct ColorCorrespondence;
struct ColorBalanceResult;
} // namespace opencalibration::orthomosaic

namespace opencalibration::orthomosaic
{

// RAII context for ray tracing operations - manages mesh intersection searchers
class RayTraceContext
{
  public:
    RayTraceContext() = default;
    explicit RayTraceContext(const std::vector<surface_model> &surfaces);

    // Reinitialize with new surfaces
    void init(const std::vector<surface_model> &surfaces);

    // Ray-trace to find height at world position (x, y)
    double traceHeight(double x, double y, double mean_camera_z);

    bool isValid() const
    {
        return !_searchers.empty();
    }

  private:
    std::vector<MeshIntersectionSearcher> _searchers;
};

struct OrthoMosaic
{
    GenericRaster pixelValues;
    RasterLayer<float> dsm;
    RasterLayer<uint8_t> overlap;
    RasterLayer<uint32_t> cameraUUID;
    double gsd;
};

struct OrthoMosaicBounds
{
    double min_x, max_x, min_y, max_y;
    double mean_surface_z;
};

// Context containing common data for orthomosaic generation
struct OrthoMosaicContext
{
    OrthoMosaicBounds bounds;
    double gsd;
    std::unordered_set<size_t> involved_nodes;
    jk::tree::KDTree<size_t, 2> imageGPSLocations;
    double mean_camera_z;
    double average_camera_elevation;
    RayTraceContext rayTraceContext;
};

OrthoMosaicBounds calculateBoundsAndMeanZ(const std::vector<surface_model> &surfaces);

enum class ImageResolution
{
    Thumbnail,
    FullResolution
};

double calculateGSD(const MeasurementGraph &graph, const std::unordered_set<size_t> &involved_nodes,
                    double mean_surface_z, ImageResolution resolution = ImageResolution::Thumbnail);

OrthoMosaicContext prepareOrthoMosaicContext(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                                             ImageResolution resolution = ImageResolution::Thumbnail);

// Ray-trace to find height at world position (x, y) using a context (preferred - RAII)
double rayTraceHeight(double x, double y, double mean_camera_z, RayTraceContext &context);

// Ray-trace to find height - convenience overload that creates temporary context
// Note: Less efficient for repeated calls; prefer using RayTraceContext directly
double rayTraceHeight(double x, double y, double mean_camera_z, const std::vector<surface_model> &surfaces);

struct OrthoMosaicConfig
{
    int num_layers = 3;
    int tile_size = 1024;
    int pyramid_levels = 4;
    double outlier_mad_threshold = 3.0;
    int correspondence_subsample = 10; // sample every Nth boundary pixel
};

OrthoMosaic generateOrthomosaic(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph);

void generateDSMGeoTIFF(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                        const opencalibration::GeoCoord &coord_system, const std::string &output_path,
                        int tile_size = 1024);

std::vector<ColorCorrespondence> generateLayeredGeoTIFF(const std::vector<surface_model> &surfaces,
                                                        const MeasurementGraph &graph,
                                                        const opencalibration::GeoCoord &coord_system,
                                                        const std::string &layers_path, const std::string &cameras_path,
                                                        const OrthoMosaicConfig &config = {});

void blendLayeredGeoTIFF(const std::string &layers_path, const std::string &cameras_path,
                         const std::string &output_path, const ColorBalanceResult &color_balance,
                         const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                         const opencalibration::GeoCoord &coord_system, const OrthoMosaicConfig &config = {});

} // namespace opencalibration::orthomosaic
