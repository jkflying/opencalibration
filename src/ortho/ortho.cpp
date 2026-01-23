#include <opencalibration/ortho/ortho.hpp>

#include <jk/KDTree.h>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/ortho/gdal_dataset.hpp>
#include <opencalibration/ortho/image_cache.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/surface/intersect.hpp>

#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <omp.h>

namespace
{

std::array<double, 3> to_array(const Eigen::Vector3d &v)
{
    return {v.x(), v.y(), v.z()};
}

Eigen::Vector2i size(const opencalibration::GenericRaster &raster)
{
    const auto getSize = [](const auto &rasterInstance) -> Eigen::Vector2i {
        return {rasterInstance.layers[0].pixels.rows(), rasterInstance.layers[0].pixels.cols()};
    };
    return std::visit(getSize, raster);
}

} // namespace

namespace opencalibration::orthomosaic
{

OrthoMosaicBounds calculateBoundsAndMeanZ(const std::vector<surface_model> &surfaces)
{
    const double inf = std::numeric_limits<double>::infinity();
    double min_x = inf, min_y = inf, max_x = -inf, max_y = -inf;
    double mean_surface_z = 0;
    size_t count_z = 0;

    for (const auto &surface : surfaces)
    {
        bool has_mesh = false;
        double s_min_x = inf, s_max_x = -inf, s_min_y = inf, s_max_y = -inf;
        for (auto iter = surface.mesh.cnodebegin(); iter != surface.mesh.cnodeend(); ++iter)
        {
            const auto &loc = iter->second.payload.location;

            mean_surface_z = (loc.z() + mean_surface_z * count_z) / (count_z + 1);
            s_min_x = std::min(s_min_x, loc.x());
            s_max_x = std::max(s_max_x, loc.x());
            s_min_y = std::min(s_min_y, loc.y());
            s_max_y = std::max(s_max_y, loc.y());
            count_z++;

            has_mesh = true;
        }

        if (!has_mesh)
            for (const auto &points : surface.cloud)
            {
                for (const auto &loc : points)
                {
                    mean_surface_z = (loc.z() + mean_surface_z * count_z) / (count_z + 1);
                    s_min_x = std::min(s_min_x, loc.x());
                    s_max_x = std::max(s_max_x, loc.x());
                    s_min_y = std::min(s_min_y, loc.y());
                    s_max_y = std::max(s_max_y, loc.y());
                    count_z++;
                }
            }

        min_x = std::min(min_x, s_min_x);
        max_x = std::max(max_x, s_max_x);
        min_y = std::min(min_y, s_min_y);
        max_y = std::max(max_y, s_max_y);
    }

    return {min_x, max_x, min_y, max_y, mean_surface_z};
}

double calculateGSD(const MeasurementGraph &graph, const std::unordered_set<size_t> &involved_nodes,
                    double mean_surface_z)
{
    double thumb_arc_pixel = 0;
    double mean_camera_z = 0;
    size_t thumb_count = 0;

    for (size_t node_id : involved_nodes)
    {
        const auto *node = graph.getNode(node_id);
        if (!node)
            continue;
        const auto &payload = node->payload;
        const double h = 0.001;
        Eigen::Vector2d pixel = image_from_3d({0, 0, 1}, *payload.model);
        Eigen::Vector2d pixelShift = image_from_3d({h, 0, 1}, *payload.model);
        double arc_pixel = h / (pixel - pixelShift).norm();
        double thumb_scale = 1.0;

        if (payload.model->pixels_rows > 0)
        {
            thumb_scale = static_cast<double>(size(payload.thumbnail)[0]) / payload.model->pixels_rows;
        }

        thumb_arc_pixel = (thumb_arc_pixel * thumb_count + arc_pixel / thumb_scale) / (thumb_count + 1);
        mean_camera_z = (mean_camera_z * thumb_count + payload.position.z()) / (thumb_count + 1);
        thumb_count++;
    }

    const double average_camera_elevation = mean_camera_z - mean_surface_z;
    double mean_gsd = std::abs(average_camera_elevation * thumb_arc_pixel);
    mean_gsd = std::max(mean_gsd, 0.001);
    return mean_gsd;
}

OrthoMosaicContext prepareOrthoMosaicContext(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph)
{
    OrthoMosaicContext context;

    // Calculate bounds
    context.bounds = calculateBoundsAndMeanZ(surfaces);

    // Collect involved nodes (nodes with finite orientation)
    for (auto iter = graph.cnodebegin(); iter != graph.cnodeend(); ++iter)
    {
        if (iter->second.payload.orientation.coeffs().allFinite())
        {
            context.involved_nodes.insert(iter->first);
        }
    }

    // Calculate GSD
    context.gsd = calculateGSD(graph, context.involved_nodes, context.bounds.mean_surface_z);

    // Build KDTree and calculate mean camera Z
    context.mean_camera_z = 0;
    size_t count = 0;
    for (size_t node_id : context.involved_nodes)
    {
        const auto *node = graph.getNode(node_id);
        context.imageGPSLocations.addPoint(to_array(node->payload.position), node_id, false);
        context.mean_camera_z = (context.mean_camera_z * count + node->payload.position.z()) / (count + 1);
        count++;
    }
    context.imageGPSLocations.splitOutstanding();

    context.average_camera_elevation = context.mean_camera_z - context.bounds.mean_surface_z;

    // Initialize ray trace context
    context.rayTraceContext.init(surfaces);

    return context;
}

// RayTraceContext implementation
RayTraceContext::RayTraceContext(const std::vector<surface_model> &surfaces)
{
    init(surfaces);
}

void RayTraceContext::init(const std::vector<surface_model> &surfaces)
{
    _searchers.clear();
    for (const auto &surface : surfaces)
    {
        _searchers.emplace_back();
        if (!_searchers.back().init(surface.mesh))
        {
            _searchers.pop_back();
        }
    }
}

double RayTraceContext::traceHeight(double x, double y, double mean_camera_z)
{
    const ray_d intersectionRay{{0, 0, -1}, {x, y, mean_camera_z}};

    for (auto &searcher : _searchers)
    {
        if (searcher.lastResult().type != MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
        {
            if (!searcher.reinit())
            {
                continue;
            }
        }

        auto intersection = searcher.triangleIntersect(intersectionRay);
        if (intersection.type == MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
        {
            return intersection.intersectionLocation.z();
        }
    }

    return NAN;
}

double rayTraceHeight(double x, double y, double mean_camera_z, RayTraceContext &context)
{
    return context.traceHeight(x, y, mean_camera_z);
}

double rayTraceHeight(double x, double y, double mean_camera_z, const std::vector<surface_model> &surfaces)
{
    // Convenience overload - creates temporary context for simple use cases
    RayTraceContext context(surfaces);
    return context.traceHeight(x, y, mean_camera_z);
}

OrthoMosaic generateOrthomosaic(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph)
{
    // Prepare common context
    OrthoMosaicContext context = prepareOrthoMosaicContext(surfaces, graph);

    spdlog::info("x range [{}; {}]  y range [{}; {}]  mean surface {}", context.bounds.min_x, context.bounds.max_x,
                 context.bounds.min_y, context.bounds.max_y, context.bounds.mean_surface_z);

    // from bounds and gsd, calculate image resolution
    double image_width = (context.bounds.max_x - context.bounds.min_x) / context.gsd;
    double image_height = (context.bounds.max_y - context.bounds.min_y) / context.gsd;

    if (!std::isfinite(image_width) || image_width < 1)
        image_width = 100;
    if (!std::isfinite(image_height) || image_height < 1)
        image_height = 100;

    spdlog::info("requested image_width: {} image_height: {}", image_width, image_height);
    spdlog::info("max_x: {} min_x: {} max_y: {} min_y: {}", context.bounds.max_x, context.bounds.min_x,
                 context.bounds.max_y, context.bounds.min_y);

    cv::Size image_dimensions(static_cast<int>(image_width), static_cast<int>(image_height));

    spdlog::info("gsd {}  img dims {}x{}", context.gsd, image_dimensions.width, image_dimensions.height);

    OrthoMosaic result;
    result.gsd = context.gsd;
    MultiLayerRaster<uint8_t> pixelValues(image_dimensions.height, image_dimensions.width, 4);
    // result.pixelValues; // assign at end
    result.cameraUUID.pixels.resize(image_dimensions.height, image_dimensions.width);
    result.overlap.pixels.resize(image_dimensions.height, image_dimensions.width);
    result.dsm.pixels.resize(image_dimensions.height, image_dimensions.width);

    PerformanceMeasure p("Generate thumbnail");

    // Precompute inverse rotation matrices to avoid quaternion.inverse() per pixel
    struct CameraCache
    {
        Eigen::Matrix3d inv_rotation;
        double thumb_scale;
        Eigen::Vector2i thumb_size;
    };
    std::unordered_map<size_t, CameraCache> camera_cache;
    for (size_t node_id : context.involved_nodes)
    {
        const auto *node = graph.getNode(node_id);
        const auto &payload = node->payload;
        CameraCache cc;
        cc.inv_rotation = payload.orientation.inverse().toRotationMatrix();
        Eigen::Vector2i sz = size(payload.thumbnail);
        cc.thumb_size = sz;
        cc.thumb_scale = payload.model->pixels_rows > 0 ? static_cast<double>(sz[0]) / payload.model->pixels_rows : 1.0;
        camera_cache[node_id] = cc;
    }

    std::atomic<int> completed_rows{0};
    auto last_log_time = std::chrono::steady_clock::now();

#pragma omp parallel
    {
        std::vector<MeshIntersectionSearcher> searchers;
        for (const auto &surface : surfaces)
        {
            searchers.emplace_back();
            if (!searchers.back().init(surface.mesh))
            {
                spdlog::error("Could not initialize searcher on mesh surface");
                searchers.pop_back();
            }
        }

#pragma omp for schedule(dynamic)
        for (int row = 0; row < image_dimensions.height; row++)
        {
            for (int col = 0; col < image_dimensions.width; col++)
            {
                const double x = col * context.gsd + context.bounds.min_x;
                const double y = context.bounds.max_y - row * context.gsd;

                // get height of pixel from mesh or nearest keypoint
                const ray_d intersectionRay{{0, 0, -1}, {x, y, context.mean_camera_z}};
                double z = NAN;
                for (auto &searcher : searchers)
                {
                    if (searcher.lastResult().type != MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                    {
                        if (!searcher.reinit())
                        {
                            continue;
                        }
                    }

                    auto intersection = searcher.triangleIntersect(intersectionRay);
                    if (intersection.type == MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                    {
                        z = intersection.intersectionLocation.z();
                        break;
                    }
                }

                if (std::isnan(z))
                {
                    continue;
                }

                Eigen::Vector3d sample_point(x, y, z);

                Eigen::Vector<uint8_t, Eigen::Dynamic> color;
                color.resize(4);
                color.fill(0);
                uint32_t pixelSource = std::numeric_limits<uint32_t>::max();

                auto closest5 = context.imageGPSLocations.searchKnn({x, y, context.average_camera_elevation}, 5);

                // get image vertically closest
                for (const auto &closest : closest5)
                {
                    const auto *closestNode = graph.getNode(closest.payload);
                    const auto &payload = closestNode->payload;
                    const auto &cc = camera_cache.at(closest.payload);

                    // Use overload with precomputed inverse rotation matrix
                    Eigen::Vector2d pixel =
                        image_from_3d(sample_point, *payload.model, payload.position, cc.inv_rotation);
                    Eigen::Vector2d thumb_pixel = pixel * cc.thumb_scale;

                    int px = static_cast<int>(thumb_pixel.x());
                    int py = static_cast<int>(thumb_pixel.y());

                    if (px > 0 && px < cc.thumb_size[1] && py > 0 && py < cc.thumb_size[0])
                    {
                        Eigen::Vector<uint8_t, Eigen::Dynamic> pixelValue;
                        pixelValue.resize(3);
                        if (payload.thumbnail.get(py, px, pixelValue))
                        {
                            color << pixelValue, 255;
                            pixelSource = closest.payload & 0xFFFFFFFF;
                            break;
                        }
                    }
                }

                if (pixelSource == std::numeric_limits<uint32_t>::max())
                {
                    // background checkerboard
                    uint8_t grey = (row + col) % 2 == 0 ? 64 : 128;
                    color << grey, grey, grey, 0; // alpha 0 for background
                }

                // assign color to thumbnail pixel
                pixelValues.set(row, col, color);
                result.cameraUUID.pixels(row, col) = pixelSource;
            }

            int current_completed = ++completed_rows;
            if (omp_get_thread_num() == 0)
            {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() >= 5)
                {
                    spdlog::info("Thumbnail generation progress: {:.1f}%",
                                 100.0 * current_completed / image_dimensions.height);
                    last_log_time = now;
                }
            }
        }
    }

    result.pixelValues = std::move(pixelValues);
    return result;

    //     cv::imwrite("thumbnail.png", image);
    //     cv::imwrite("source.png", source);

    // TODO: some kind of color balancing of the different patches - maybe in LAB space?
    // TODO: laplacian or gradient domain blending of the differrent patches
}

namespace
{
// Helper: Create GDAL GeoTIFF dataset
GDALDatasetPtr createGeoTIFF(const std::string &path, int width, int height, double min_x, double max_y, double gsd,
                             const std::string &wkt)
{
    GDALAllRegister();

    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver)
    {
        throw std::runtime_error("GTiff driver not available");
    }

    char **options = nullptr;
    options = CSLSetNameValue(options, "TILED", "YES");
    options = CSLSetNameValue(options, "BLOCKXSIZE", "512");
    options = CSLSetNameValue(options, "BLOCKYSIZE", "512");
    options = CSLSetNameValue(options, "COMPRESS", "LZW");
    options = CSLSetNameValue(options, "PHOTOMETRIC", "RGB");
    options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");

    GDALDataset *dataset = driver->Create(path.c_str(), width, height, 4, GDT_Byte, options);

    CSLDestroy(options);

    if (!dataset)
    {
        throw std::runtime_error("Failed to create GeoTIFF: " + path);
    }

    // Set geotransform: [min_x, gsd, 0, max_y, 0, -gsd]
    double geotransform[6] = {min_x, gsd, 0, max_y, 0, -gsd};
    dataset->SetGeoTransform(geotransform);

    // Set projection
    if (!wkt.empty())
    {
        dataset->SetProjection(wkt.c_str());
    }

    // Set band interpretation
    dataset->GetRasterBand(1)->SetColorInterpretation(GCI_RedBand);
    dataset->GetRasterBand(2)->SetColorInterpretation(GCI_GreenBand);
    dataset->GetRasterBand(3)->SetColorInterpretation(GCI_BlueBand);
    dataset->GetRasterBand(4)->SetColorInterpretation(GCI_AlphaBand);

    return GDALDatasetPtr(dataset);
}

// Helper: Write tile data to GeoTIFF
void writeTileToGeoTIFF(GDALDataset *dataset, int x_offset, int y_offset, int width, int height,
                        const std::vector<uint8_t> &buffer)
{
    // Deinterleave RGBA -> separate bands
    std::vector<uint8_t> band_data[4];
    for (int i = 0; i < 4; i++)
    {
        band_data[i].resize(width * height);
    }

    for (int i = 0; i < width * height; i++)
    {
        band_data[0][i] = buffer[i * 4 + 0]; // R
        band_data[1][i] = buffer[i * 4 + 1]; // G
        band_data[2][i] = buffer[i * 4 + 2]; // B
        band_data[3][i] = buffer[i * 4 + 3]; // A
    }

    // Write each band
    for (int band = 1; band <= 4; band++)
    {
        CPLErr err = dataset->GetRasterBand(band)->RasterIO(GF_Write, x_offset, y_offset, width, height,
                                                            band_data[band - 1].data(), width, height, GDT_Byte, 0, 0);

        if (err != CE_None)
        {
            throw std::runtime_error("Failed to write tile to GeoTIFF");
        }
    }
}

// Helper: Determine which images contribute to a tile
std::vector<size_t> getContributingImages(double tile_min_x, double tile_max_x, double tile_min_y, double tile_max_y,
                                          double mean_surface_z, const MeasurementGraph &graph,
                                          const std::unordered_set<size_t> &involved_nodes)
{
    std::vector<size_t> contributing;

    for (size_t node_id : involved_nodes)
    {
        const auto *node = graph.getNode(node_id);
        if (!node)
            continue;

        // Calculate tile center
        double tile_center_x = (tile_min_x + tile_max_x) * 0.5;
        double tile_center_y = (tile_min_y + tile_max_y) * 0.5;

        // Simple distance-based culling: check if camera is reasonably close to tile
        Eigen::Vector2d camera_pos_2d = node->payload.position.head<2>();
        Eigen::Vector2d tile_center_2d(tile_center_x, tile_center_y);
        double dist_to_tile = (camera_pos_2d - tile_center_2d).norm();

        // Conservative estimate: 5x the tile diagonal
        double tile_diagonal = std::sqrt(std::pow(tile_max_x - tile_min_x, 2) + std::pow(tile_max_y - tile_min_y, 2));
        double max_distance = 5.0 * tile_diagonal;

        if (dist_to_tile > max_distance)
        {
            continue;
        }

        // Check if any of the tile corners project into the camera image
        std::vector<Eigen::Vector3d> corners = {{tile_min_x, tile_min_y, mean_surface_z},
                                                {tile_max_x, tile_min_y, mean_surface_z},
                                                {tile_min_x, tile_max_y, mean_surface_z},
                                                {tile_max_x, tile_max_y, mean_surface_z},
                                                {tile_center_x, tile_center_y, mean_surface_z}};

        bool can_see_tile = false;
        for (const auto &corner : corners)
        {
            Eigen::Vector2d pixel =
                image_from_3d(corner, *node->payload.model, node->payload.position, node->payload.orientation);

            if (pixel.x() >= 0 && pixel.x() < node->payload.model->pixels_cols && pixel.y() >= 0 &&
                pixel.y() < node->payload.model->pixels_rows)
            {
                can_see_tile = true;
                break;
            }
        }

        if (can_see_tile)
        {
            contributing.push_back(node_id);
        }
    }

    return contributing;
}

// Helper: Process a single tile
void processTile(GDALDataset *dataset, int tile_x, int tile_y, int tile_size, const OrthoMosaicBounds &bounds,
                 double gsd, int output_width, int output_height, const std::vector<surface_model> &surfaces,
                 const MeasurementGraph &graph, const std::unordered_set<size_t> &involved_nodes,
                 const jk::tree::KDTree<size_t, 3> &imageGPSLocations, double average_camera_elevation,
                 double mean_camera_z, FullResolutionImageCache &image_cache)
{
    // Calculate tile bounds in pixels
    int x_offset = tile_x * tile_size;
    int y_offset = tile_y * tile_size;
    int tile_width = std::min(tile_size, output_width - x_offset);
    int tile_height = std::min(tile_size, output_height - y_offset);

    // Calculate tile bounds in world coordinates
    double tile_min_x = bounds.min_x + x_offset * gsd;
    double tile_max_x = bounds.min_x + (x_offset + tile_width) * gsd;
    double tile_min_y = bounds.max_y - (y_offset + tile_height) * gsd;
    double tile_max_y = bounds.max_y - y_offset * gsd;

    // Determine contributing images
    std::vector<size_t> contributing = getContributingImages(tile_min_x, tile_max_x, tile_min_y, tile_max_y,
                                                             bounds.mean_surface_z, graph, involved_nodes);

    spdlog::debug("Tile ({}, {}): {} contributing images", tile_x, tile_y, contributing.size());

    // Allocate tile buffer
    std::vector<uint8_t> tile_buffer(tile_width * tile_height * 4);

    // Process pixels in tile
#pragma omp parallel
    {
        std::vector<MeshIntersectionSearcher> searchers;
        for (const auto &surface : surfaces)
        {
            searchers.emplace_back();
            if (!searchers.back().init(surface.mesh))
            {
                spdlog::error("Could not initialize searcher on mesh surface");
                searchers.pop_back();
            }
        }

#pragma omp for schedule(dynamic)
        for (int local_row = 0; local_row < tile_height; local_row++)
        {
            for (int local_col = 0; local_col < tile_width; local_col++)
            {
                int global_col = x_offset + local_col;
                int global_row = y_offset + local_row;

                // World coordinates
                const double x = global_col * gsd + bounds.min_x;
                const double y = bounds.max_y - global_row * gsd;

                // Ray-trace to get height
                const ray_d intersectionRay{{0, 0, -1}, {x, y, mean_camera_z}};
                double z = NAN;
                for (auto &searcher : searchers)
                {
                    if (searcher.lastResult().type != MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                    {
                        if (!searcher.reinit())
                        {
                            continue;
                        }
                    }

                    auto intersection = searcher.triangleIntersect(intersectionRay);
                    if (intersection.type == MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                    {
                        z = intersection.intersectionLocation.z();
                        break;
                    }
                }

                int idx = (local_row * tile_width + local_col) * 4;

                if (std::isnan(z))
                {
                    // Background checkerboard
                    uint8_t grey = (global_row + global_col) % 2 == 0 ? 64 : 128;
                    tile_buffer[idx + 0] = grey;
                    tile_buffer[idx + 1] = grey;
                    tile_buffer[idx + 2] = grey;
                    tile_buffer[idx + 3] = 0;
                    continue;
                }

                Eigen::Vector3d sample_point(x, y, z);

                // Find best image and sample color
                auto closest5 = imageGPSLocations.searchKnn({x, y, average_camera_elevation}, 5);

                bool found_color = false;
                for (const auto &closest : closest5)
                {
                    const auto *closestNode = graph.getNode(closest.payload);
                    const auto &payload = closestNode->payload;

                    // Backproject 3D point to image space
                    Eigen::Vector2d pixel =
                        image_from_3d(sample_point, *payload.model, payload.position, payload.orientation);

                    // Check if pixel is within image bounds
                    if (pixel.x() < 0 || pixel.x() >= payload.model->pixels_cols || pixel.y() < 0 ||
                        pixel.y() >= payload.model->pixels_rows)
                    {
                        continue;
                    }

                    // Load full-resolution image
                    cv::Mat full_image = image_cache.getImage(closest.payload, payload.path);

                    if (full_image.empty())
                    {
                        continue;
                    }

                    // Sample color at pixel location
                    int px = static_cast<int>(pixel.x());
                    int py = static_cast<int>(pixel.y());

                    if (px >= 0 && px < full_image.cols && py >= 0 && py < full_image.rows)
                    {
                        cv::Vec3b color_bgr = full_image.at<cv::Vec3b>(py, px);

                        // Convert BGR to RGB
                        tile_buffer[idx + 0] = color_bgr[2]; // R
                        tile_buffer[idx + 1] = color_bgr[1]; // G
                        tile_buffer[idx + 2] = color_bgr[0]; // B
                        tile_buffer[idx + 3] = 255;          // A

                        found_color = true;
                        break;
                    }
                }

                if (!found_color)
                {
                    // Background checkerboard
                    uint8_t grey = (global_row + global_col) % 2 == 0 ? 64 : 128;
                    tile_buffer[idx + 0] = grey;
                    tile_buffer[idx + 1] = grey;
                    tile_buffer[idx + 2] = grey;
                    tile_buffer[idx + 3] = 0;
                }
            }
        }
    }

    // Write tile to GeoTIFF
    writeTileToGeoTIFF(dataset, x_offset, y_offset, tile_width, tile_height, tile_buffer);
}

} // namespace

void generateGeoTIFFOrthomosaic(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                                const opencalibration::GeoCoord &coord_system, const std::string &output_path,
                                int tile_size)
{
    spdlog::info("Generating full-resolution GeoTIFF orthomosaic: {}", output_path);

    PerformanceMeasure p("Generate GeoTIFF orthomosaic");

    // Prepare common context
    OrthoMosaicContext context = prepareOrthoMosaicContext(surfaces, graph);

    spdlog::info("x range [{}; {}]  y range [{}; {}]  mean surface {}", context.bounds.min_x, context.bounds.max_x,
                 context.bounds.min_y, context.bounds.max_y, context.bounds.mean_surface_z);

    // Calculate output dimensions
    int width = static_cast<int>((context.bounds.max_x - context.bounds.min_x) / context.gsd);
    int height = static_cast<int>((context.bounds.max_y - context.bounds.min_y) / context.gsd);

    if (width <= 0)
        width = 100;
    if (height <= 0)
        height = 100;

    spdlog::info("GSD: {}  Output dimensions: {}x{} pixels", context.gsd, width, height);

    // Create GeoTIFF dataset
    std::string wkt = coord_system.getWKT();
    GDALDatasetPtr dataset =
        createGeoTIFF(output_path, width, height, context.bounds.min_x, context.bounds.max_y, context.gsd, wkt);

    // Create image cache
    FullResolutionImageCache image_cache(10);

    // Process tiles
    int num_tiles_x = (width + tile_size - 1) / tile_size;
    int num_tiles_y = (height + tile_size - 1) / tile_size;
    int total_tiles = num_tiles_x * num_tiles_y;

    spdlog::info("Processing {} tiles ({}x{} grid, tile size: {}x{})", total_tiles, num_tiles_x, num_tiles_y, tile_size,
                 tile_size);

    int completed_tiles = 0;
    auto start_time = std::chrono::steady_clock::now();

    for (int tile_y = 0; tile_y < num_tiles_y; tile_y++)
    {
        for (int tile_x = 0; tile_x < num_tiles_x; tile_x++)
        {
            processTile(dataset.get(), tile_x, tile_y, tile_size, context.bounds, context.gsd, width, height, surfaces,
                        graph, context.involved_nodes, context.imageGPSLocations, context.average_camera_elevation,
                        context.mean_camera_z, image_cache);

            // Clear cache after each tile to bound memory
            image_cache.clear();

            completed_tiles++;

            if (completed_tiles % 10 == 0 || completed_tiles == total_tiles)
            {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                double progress = 100.0 * completed_tiles / total_tiles;
                spdlog::info("Progress: {:.1f}% ({}/{} tiles, {} seconds)", progress, completed_tiles, total_tiles,
                             elapsed);
            }
        }
    }

    spdlog::info("GeoTIFF orthomosaic generation complete: {}", output_path);
}

namespace
{
// Helper: Create GDAL GeoTIFF dataset for DSM (single float32 band)
GDALDatasetPtr createDSMGeoTIFF(const std::string &path, int width, int height, double min_x, double max_y, double gsd,
                                const std::string &wkt)
{
    GDALAllRegister();

    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver)
    {
        throw std::runtime_error("GTiff driver not available");
    }

    char **options = nullptr;
    options = CSLSetNameValue(options, "TILED", "YES");
    options = CSLSetNameValue(options, "BLOCKXSIZE", "512");
    options = CSLSetNameValue(options, "BLOCKYSIZE", "512");
    options = CSLSetNameValue(options, "COMPRESS", "LZW");
    options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");

    GDALDataset *dataset = driver->Create(path.c_str(), width, height, 1, GDT_Float32, options);

    CSLDestroy(options);

    if (!dataset)
    {
        throw std::runtime_error("Failed to create DSM GeoTIFF: " + path);
    }

    // Set geotransform: [min_x, gsd, 0, max_y, 0, -gsd]
    double geotransform[6] = {min_x, gsd, 0, max_y, 0, -gsd};
    dataset->SetGeoTransform(geotransform);

    // Set projection
    if (!wkt.empty())
    {
        dataset->SetProjection(wkt.c_str());
    }

    // Set nodata value for DSM
    dataset->GetRasterBand(1)->SetNoDataValue(std::numeric_limits<float>::quiet_NaN());

    return GDALDatasetPtr(dataset);
}

// Helper: Process a single tile for DSM generation
void processDSMTile(GDALDataset *dataset, int tile_x, int tile_y, int tile_size, const OrthoMosaicBounds &bounds,
                    double gsd, int output_width, int output_height, const std::vector<surface_model> &surfaces,
                    double mean_camera_z)
{
    // Calculate tile bounds in pixels
    int x_offset = tile_x * tile_size;
    int y_offset = tile_y * tile_size;
    int tile_width = std::min(tile_size, output_width - x_offset);
    int tile_height = std::min(tile_size, output_height - y_offset);

    // Allocate tile buffer
    std::vector<float> tile_buffer(tile_width * tile_height, std::numeric_limits<float>::quiet_NaN());

    // Process pixels in tile
#pragma omp parallel
    {
        std::vector<MeshIntersectionSearcher> searchers;
        for (const auto &surface : surfaces)
        {
            searchers.emplace_back();
            if (!searchers.back().init(surface.mesh))
            {
                spdlog::error("Could not initialize searcher on mesh surface");
                searchers.pop_back();
            }
        }

#pragma omp for schedule(dynamic)
        for (int local_row = 0; local_row < tile_height; local_row++)
        {
            for (int local_col = 0; local_col < tile_width; local_col++)
            {
                int global_col = x_offset + local_col;
                int global_row = y_offset + local_row;

                // World coordinates
                const double x = global_col * gsd + bounds.min_x;
                const double y = bounds.max_y - global_row * gsd;

                // Ray-trace to get height
                const ray_d intersectionRay{{0, 0, -1}, {x, y, mean_camera_z}};
                double z = NAN;
                for (auto &searcher : searchers)
                {
                    if (searcher.lastResult().type != MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                    {
                        if (!searcher.reinit())
                        {
                            continue;
                        }
                    }

                    auto intersection = searcher.triangleIntersect(intersectionRay);
                    if (intersection.type == MeshIntersectionSearcher::IntersectionInfo::INTERSECTION)
                    {
                        z = intersection.intersectionLocation.z();
                        break;
                    }
                }

                int idx = local_row * tile_width + local_col;
                tile_buffer[idx] = static_cast<float>(z);
            }
        }
    }

    // Write tile to GeoTIFF
    CPLErr err = dataset->GetRasterBand(1)->RasterIO(GF_Write, x_offset, y_offset, tile_width, tile_height,
                                                     tile_buffer.data(), tile_width, tile_height, GDT_Float32, 0, 0);

    if (err != CE_None)
    {
        throw std::runtime_error("Failed to write DSM tile to GeoTIFF");
    }
}

} // namespace

void generateDSMGeoTIFF(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                        const opencalibration::GeoCoord &coord_system, const std::string &output_path, int tile_size)
{
    spdlog::info("Generating DSM GeoTIFF: {}", output_path);

    PerformanceMeasure p("Generate DSM GeoTIFF");

    // Prepare common context
    OrthoMosaicContext context = prepareOrthoMosaicContext(surfaces, graph);

    spdlog::info("DSM: x range [{}; {}]  y range [{}; {}]  mean surface {}", context.bounds.min_x, context.bounds.max_x,
                 context.bounds.min_y, context.bounds.max_y, context.bounds.mean_surface_z);

    // Calculate output dimensions
    int width = static_cast<int>((context.bounds.max_x - context.bounds.min_x) / context.gsd);
    int height = static_cast<int>((context.bounds.max_y - context.bounds.min_y) / context.gsd);

    if (width <= 0)
        width = 100;
    if (height <= 0)
        height = 100;

    spdlog::info("DSM GSD: {}  Output dimensions: {}x{} pixels", context.gsd, width, height);

    // Create GeoTIFF dataset
    std::string wkt = coord_system.getWKT();
    GDALDatasetPtr dataset =
        createDSMGeoTIFF(output_path, width, height, context.bounds.min_x, context.bounds.max_y, context.gsd, wkt);

    // Process tiles
    int num_tiles_x = (width + tile_size - 1) / tile_size;
    int num_tiles_y = (height + tile_size - 1) / tile_size;
    int total_tiles = num_tiles_x * num_tiles_y;

    spdlog::info("DSM: Processing {} tiles ({}x{} grid, tile size: {}x{})", total_tiles, num_tiles_x, num_tiles_y,
                 tile_size, tile_size);

    int completed_tiles = 0;
    auto start_time = std::chrono::steady_clock::now();

    for (int tile_y = 0; tile_y < num_tiles_y; tile_y++)
    {
        for (int tile_x = 0; tile_x < num_tiles_x; tile_x++)
        {
            processDSMTile(dataset.get(), tile_x, tile_y, tile_size, context.bounds, context.gsd, width, height,
                           surfaces, context.mean_camera_z);

            completed_tiles++;

            if (completed_tiles % 10 == 0 || completed_tiles == total_tiles)
            {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                double progress = 100.0 * completed_tiles / total_tiles;
                spdlog::info("DSM Progress: {:.1f}% ({}/{} tiles, {} seconds)", progress, completed_tiles, total_tiles,
                             elapsed);
            }
        }
    }

    spdlog::info("DSM GeoTIFF generation complete: {}", output_path);
}

} // namespace opencalibration::orthomosaic
