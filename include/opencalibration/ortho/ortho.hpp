#pragma once

#include <opencalibration/ortho/color_balance.hpp>
#include <opencalibration/pipeline/progress.hpp>
#include <opencalibration/surface/intersect.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/raster.hpp>
#include <opencalibration/types/surface_model.hpp>

#include <jk/KDTree.h>

#include <ankerl/unordered_dense.h>
#include <string>
#include <vector>

namespace opencalibration
{
class GeoCoord;
}

namespace opencalibration::orthomosaic
{

SampleGeometry sampleGeometry(const image &payload, const Eigen::Vector3d &world_point, const Eigen::Vector2d &pixel);

class RayTraceContext
{
  public:
    explicit RayTraceContext(const std::vector<surface_model> &surfaces);

    // NaN when no surface is hit
    double traceHeight(double x, double y, double mean_camera_z);

  private:
    std::vector<MeshIntersectionSearcher> _searchers;
};

struct OrthoMosaicBounds
{
    double min_x, max_x, min_y, max_y;
    double mean_surface_z;
};

struct OrthoMosaic
{
    GenericRaster pixelValues;
    RasterLayer<float> dsm;
    RasterLayer<uint8_t> overlap;
    RasterLayer<int32_t> cameraUUID;
    double gsd;
    OrthoMosaicBounds bounds;
    ColorBalanceResult color_balance;
};

// Context containing common data for orthomosaic generation
struct OrthoMosaicContext
{
    OrthoMosaicBounds bounds;
    double gsd;
    ankerl::unordered_dense::set<size_t> involved_nodes;
    jk::tree::KDTree<size_t, 2> imageGPSLocations;
    double mean_camera_z;
    double average_camera_elevation;
};

OrthoMosaicBounds calculateBoundsAndMeanZ(const std::vector<surface_model> &surfaces);

void coarsenGsdToFit(double &gsd, int &width, int &height, const OrthoMosaicBounds &bounds, uint64_t max_pixels);

ankerl::unordered_dense::set<size_t> findTileCameras(int tile_x, int tile_y, int tile_size,
                                                     const OrthoMosaicBounds &bounds, double gsd, int output_width,
                                                     int output_height,
                                                     const jk::tree::KDTree<size_t, 2> &imageGPSLocations,
                                                     int num_neighbors);

enum class ImageResolution
{
    Thumbnail,
    FullResolution
};

double arcPerPixel(const CameraModel &model);

double calculateGSD(const MeasurementGraph &graph, const ankerl::unordered_dense::set<size_t> &involved_nodes,
                    double mean_surface_z, ImageResolution resolution = ImageResolution::Thumbnail);

OrthoMosaicContext prepareOrthoMosaicContext(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                                             ImageResolution resolution = ImageResolution::Thumbnail);

struct OrthoMosaicConfig
{
    int tile_size = 1024;
    int blend_transition_radius = 64;
    double max_output_megapixels = 0.0; // 0 = unlimited
};

OrthoMosaic generateOrthomosaic(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                                const OrthoMosaicConfig &config = {});

void generateGeoTIFF(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                     const opencalibration::GeoCoord &coord_system, const ColorBalanceResult &color_balance,
                     const std::string &output_path, const std::string &dsm_output_path,
                     const OrthoMosaicConfig &config = {}, TileProgressCallback tile_progress = {});

void generateTexturedOBJ(const std::vector<surface_model> &surfaces, const std::string &geotiff_path,
                         const std::string &obj_path);

} // namespace opencalibration::orthomosaic
