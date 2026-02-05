#include <opencalibration/ortho/ortho.hpp>

#include <ceres/jet.h>
#include <eigen3/Eigen/Eigenvalues>
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

class PatchSampler
{
  public:
    static constexpr int MAX_PATCH_RADIUS = 16;
    static constexpr int MAX_PATCH_SIZE = 2 * MAX_PATCH_RADIUS + 1;

    PatchSampler() : _lab_pixel(1, 1, CV_8UC3), _bgr_pixel(1, 1, CV_8UC3)
    {
        _patch_buffer.create(MAX_PATCH_SIZE, MAX_PATCH_SIZE, CV_8UC3);
        _lab_buffer.create(MAX_PATCH_SIZE, MAX_PATCH_SIZE, CV_8UC3);
    }

    Eigen::Matrix2d computeJacobian(const Eigen::Vector3d &world_point,
                                    const opencalibration::DifferentiableCameraModel<double> &model,
                                    const Eigen::Vector3d &camera_position,
                                    const Eigen::Matrix3d &camera_orientation_inverse) const
    {
        using JetT = ceres::Jet<double, 2>;

        Eigen::Matrix<JetT, 3, 1> world_point_jet;
        world_point_jet[0] = JetT(world_point.x(), 0);
        world_point_jet[1] = JetT(world_point.y(), 1);
        world_point_jet[2] = JetT(world_point.z());

        opencalibration::DifferentiableCameraModel<JetT> model_jet;
        model_jet.focal_length_pixels = JetT(model.focal_length_pixels);
        model_jet.principle_point = model.principle_point.cast<JetT>();
        model_jet.radial_distortion = model.radial_distortion.cast<JetT>();
        model_jet.tangential_distortion = model.tangential_distortion.cast<JetT>();
        model_jet.pixels_cols = model.pixels_cols;
        model_jet.pixels_rows = model.pixels_rows;
        model_jet.projection_type = model.projection_type;

        Eigen::Matrix<JetT, 3, 1> camera_position_jet = camera_position.cast<JetT>();
        Eigen::Matrix<JetT, 3, 3> camera_orientation_inverse_jet = camera_orientation_inverse.cast<JetT>();

        Eigen::Matrix<JetT, 2, 1> pixel_jet = opencalibration::image_from_3d(
            world_point_jet, model_jet, camera_position_jet, camera_orientation_inverse_jet);

        Eigen::Matrix2d J;
        J(0, 0) = pixel_jet[0].v[0];
        J(0, 1) = pixel_jet[0].v[1];
        J(1, 0) = pixel_jet[1].v[0];
        J(1, 1) = pixel_jet[1].v[1];

        return J;
    }

    bool sampleWithJacobian(const cv::Mat &bgr_image, const Eigen::Vector3d &world_point,
                            const opencalibration::DifferentiableCameraModel<double> &model,
                            const Eigen::Vector3d &camera_position, const Eigen::Matrix3d &camera_orientation_inverse,
                            double output_gsd, cv::Vec3b &result)
    {
        Eigen::Vector2d pixel =
            opencalibration::image_from_3d(world_point, model, camera_position, camera_orientation_inverse);

        if (pixel.x() < 0 || pixel.x() >= bgr_image.cols || pixel.y() < 0 || pixel.y() >= bgr_image.rows)
        {
            return false;
        }

        Eigen::Matrix2d J = computeJacobian(world_point, model, camera_position, camera_orientation_inverse);

        // World-space gsd*gsd square maps to pixel-space ellipse: M = gsd^2 * J * J^T
        Eigen::Matrix2d M = output_gsd * output_gsd * J * J.transpose();

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(M);
        Eigen::Vector2d eigenvalues = solver.eigenvalues();

        double a = std::sqrt(std::max(eigenvalues(1), 1e-6));
        double b = std::sqrt(std::max(eigenvalues(0), 1e-6));

        if (a < 1.0 && b < 1.0)
        {
            int px = static_cast<int>(pixel.x());
            int py = static_cast<int>(pixel.y());
            if (px >= 0 && px < bgr_image.cols && py >= 0 && py < bgr_image.rows)
            {
                result = bgr_image.at<cv::Vec3b>(py, px);
                return true;
            }
            return false;
        }

        int radius = std::min(static_cast<int>(std::ceil(a)), MAX_PATCH_RADIUS);

        int x_min = std::max(0, static_cast<int>(pixel.x()) - radius);
        int y_min = std::max(0, static_cast<int>(pixel.y()) - radius);
        int x_max = std::min(bgr_image.cols - 1, static_cast<int>(pixel.x()) + radius);
        int y_max = std::min(bgr_image.rows - 1, static_cast<int>(pixel.y()) + radius);

        if (x_min > x_max || y_min > y_max)
        {
            return false;
        }

        double det = M.determinant();
        if (det < 1e-12)
        {
            int px = static_cast<int>(pixel.x());
            int py = static_cast<int>(pixel.y());
            result = bgr_image.at<cv::Vec3b>(py, px);
            return true;
        }
        Eigen::Matrix2d M_inv = M.inverse();

        double sum_L = 0, sum_a = 0, sum_b = 0;
        int count = 0;

        for (int py = y_min; py <= y_max; py++)
        {
            for (int px = x_min; px <= x_max; px++)
            {
                Eigen::Vector2d diff(px - pixel.x(), py - pixel.y());
                double ellipse_dist = diff.transpose() * M_inv * diff;

                if (ellipse_dist <= 1.0)
                {
                    cv::Vec3b bgr = bgr_image.at<cv::Vec3b>(py, px);

                    _lab_pixel.at<cv::Vec3b>(0, 0) = bgr;
                    cv::cvtColor(_lab_pixel, _bgr_pixel, cv::COLOR_BGR2Lab);
                    cv::Vec3b lab = _bgr_pixel.at<cv::Vec3b>(0, 0);

                    sum_L += lab[0];
                    sum_a += lab[1];
                    sum_b += lab[2];
                    count++;
                }
            }
        }

        if (count == 0)
        {
            int px = static_cast<int>(pixel.x());
            int py = static_cast<int>(pixel.y());
            result = bgr_image.at<cv::Vec3b>(py, px);
            return true;
        }

        _lab_pixel.at<cv::Vec3b>(0, 0) =
            cv::Vec3b(static_cast<uint8_t>(sum_L / count), static_cast<uint8_t>(sum_a / count),
                      static_cast<uint8_t>(sum_b / count));
        cv::cvtColor(_lab_pixel, _bgr_pixel, cv::COLOR_Lab2BGR);
        result = _bgr_pixel.at<cv::Vec3b>(0, 0);

        return true;
    }

  private:
    cv::Mat _patch_buffer;
    cv::Mat _lab_buffer;
    mutable cv::Mat _lab_pixel;
    mutable cv::Mat _bgr_pixel;
};

} // namespace

namespace opencalibration::orthomosaic
{

OrthoMosaicBounds calculateBoundsAndMeanZ(const std::vector<surface_model> &surfaces)
{
    const double inf = std::numeric_limits<double>::infinity();
    double min_x = inf, min_y = inf, max_x = -inf, max_y = -inf;
    std::vector<double> z_values;

    for (const auto &surface : surfaces)
    {
        bool has_mesh = false;
        double s_min_x = inf, s_max_x = -inf, s_min_y = inf, s_max_y = -inf;
        for (auto iter = surface.mesh.cnodebegin(); iter != surface.mesh.cnodeend(); ++iter)
        {
            const auto &loc = iter->second.payload.location;

            if (std::isfinite(loc.z()))
            {
                z_values.push_back(loc.z());
            }
            s_min_x = std::min(s_min_x, loc.x());
            s_max_x = std::max(s_max_x, loc.x());
            s_min_y = std::min(s_min_y, loc.y());
            s_max_y = std::max(s_max_y, loc.y());

            has_mesh = true;
        }

        if (!has_mesh)
            for (const auto &points : surface.cloud)
            {
                for (const auto &loc : points)
                {
                    if (std::isfinite(loc.z()))
                    {
                        z_values.push_back(loc.z());
                    }
                    s_min_x = std::min(s_min_x, loc.x());
                    s_max_x = std::max(s_max_x, loc.x());
                    s_min_y = std::min(s_min_y, loc.y());
                    s_max_y = std::max(s_max_y, loc.y());
                }
            }

        min_x = std::min(min_x, s_min_x);
        max_x = std::max(max_x, s_max_x);
        min_y = std::min(min_y, s_min_y);
        max_y = std::max(max_y, s_max_y);
    }

    double mean_surface_z = 0;
    for (double z : z_values)
    {
        mean_surface_z += z;
    }
    if (!z_values.empty())
    {
        mean_surface_z /= z_values.size();
    }

    return {min_x, max_x, min_y, max_y, mean_surface_z};
}

double calculateGSD(const MeasurementGraph &graph, const std::unordered_set<size_t> &involved_nodes,
                    double mean_surface_z, ImageResolution resolution)
{
    double arc_per_pixel = 0;
    double mean_camera_z = 0;
    size_t count = 0;

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

        if (resolution == ImageResolution::Thumbnail && payload.model->pixels_rows > 0)
        {
            double thumb_scale = static_cast<double>(size(payload.thumbnail)[0]) / payload.model->pixels_rows;
            arc_pixel = arc_pixel / thumb_scale;
        }

        arc_per_pixel = (arc_per_pixel * count + arc_pixel) / (count + 1);
        mean_camera_z = (mean_camera_z * count + payload.position.z()) / (count + 1);
        count++;
    }

    double average_camera_elevation = mean_camera_z - mean_surface_z;
    double mean_gsd = std::abs(average_camera_elevation * arc_per_pixel);
    mean_gsd = std::max(mean_gsd, 0.001);
    return mean_gsd;
}

OrthoMosaicContext prepareOrthoMosaicContext(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                                             ImageResolution resolution)
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
    context.gsd = calculateGSD(graph, context.involved_nodes, context.bounds.mean_surface_z, resolution);

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
    OrthoMosaicContext context = prepareOrthoMosaicContext(surfaces, graph, ImageResolution::Thumbnail);

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

void processTile(GDALDataset *dataset, int tile_x, int tile_y, int tile_size, const OrthoMosaicBounds &bounds,
                 double gsd, int output_width, int output_height, const std::vector<surface_model> &surfaces,
                 const MeasurementGraph &graph, const jk::tree::KDTree<size_t, 3> &imageGPSLocations,
                 double average_camera_elevation, double mean_camera_z,
                 const std::unordered_map<size_t, Eigen::Matrix3d> &inv_rotation_cache,
                 FullResolutionImageCache &image_cache)
{
    int x_offset = tile_x * tile_size;
    int y_offset = tile_y * tile_size;
    int tile_width = std::min(tile_size, output_width - x_offset);
    int tile_height = std::min(tile_size, output_height - y_offset);

    std::vector<uint8_t> tile_buffer(tile_width * tile_height * 4);

#pragma omp parallel
    {
        PerformanceMeasure thread_perf("Ortho tile rows");

        PatchSampler sampler;

        std::vector<MeshIntersectionSearcher> searchers;
        for (const auto &surface : surfaces)
        {
            searchers.emplace_back();
            if (!searchers.back().init(surface.mesh))
            {
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

                const double x = global_col * gsd + bounds.min_x;
                const double y = bounds.max_y - global_row * gsd;

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
                    uint8_t grey = (global_row + global_col) % 2 == 0 ? 64 : 128;
                    tile_buffer[idx + 0] = grey;
                    tile_buffer[idx + 1] = grey;
                    tile_buffer[idx + 2] = grey;
                    tile_buffer[idx + 3] = 0;
                    continue;
                }

                Eigen::Vector3d sample_point(x, y, z);

                auto closest5 = imageGPSLocations.searchKnn({x, y, average_camera_elevation}, 5);

                bool found_color = false;
                for (const auto &closest : closest5)
                {
                    const auto *closestNode = graph.getNode(closest.payload);
                    const auto &payload = closestNode->payload;

                    auto inv_rot_it = inv_rotation_cache.find(closest.payload);
                    if (inv_rot_it == inv_rotation_cache.end())
                    {
                        continue;
                    }
                    const Eigen::Matrix3d &inv_rotation = inv_rot_it->second;

                    Eigen::Vector2d pixel = image_from_3d(sample_point, *payload.model, payload.position, inv_rotation);

                    if (pixel.x() < 0 || pixel.x() >= payload.model->pixels_cols || pixel.y() < 0 ||
                        pixel.y() >= payload.model->pixels_rows)
                    {
                        continue;
                    }

                    cv::Mat full_image = image_cache.getImage(closest.payload, payload.path);

                    if (full_image.empty())
                    {
                        continue;
                    }

                    cv::Vec3b color_bgr;
                    if (sampler.sampleWithJacobian(full_image, sample_point, *payload.model, payload.position,
                                                   inv_rotation, gsd, color_bgr))
                    {
                        tile_buffer[idx + 0] = color_bgr[2];
                        tile_buffer[idx + 1] = color_bgr[1];
                        tile_buffer[idx + 2] = color_bgr[0];
                        tile_buffer[idx + 3] = 255;

                        found_color = true;
                        break;
                    }
                }

                if (!found_color)
                {
                    uint8_t grey = (global_row + global_col) % 2 == 0 ? 64 : 128;
                    tile_buffer[idx + 0] = grey;
                    tile_buffer[idx + 1] = grey;
                    tile_buffer[idx + 2] = grey;
                    tile_buffer[idx + 3] = 0;
                }
            }
        }
    }

    writeTileToGeoTIFF(dataset, x_offset, y_offset, tile_width, tile_height, tile_buffer);
}

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

void processDSMTile(GDALDataset *dataset, int tile_x, int tile_y, int tile_size, const OrthoMosaicBounds &bounds,
                    double gsd, int output_width, int output_height, const std::vector<surface_model> &surfaces,
                    double mean_camera_z)
{
    int x_offset = tile_x * tile_size;
    int y_offset = tile_y * tile_size;
    int tile_width = std::min(tile_size, output_width - x_offset);
    int tile_height = std::min(tile_size, output_height - y_offset);

    std::vector<float> tile_buffer(tile_width * tile_height, std::numeric_limits<float>::quiet_NaN());

#pragma omp parallel
    {
        PerformanceMeasure thread_perf("DSM tile rows");

        std::vector<MeshIntersectionSearcher> searchers;
        for (const auto &surface : surfaces)
        {
            searchers.emplace_back();
            if (!searchers.back().init(surface.mesh))
            {
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

                const double x = global_col * gsd + bounds.min_x;
                const double y = bounds.max_y - global_row * gsd;

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

    CPLErr err = dataset->GetRasterBand(1)->RasterIO(GF_Write, x_offset, y_offset, tile_width, tile_height,
                                                     tile_buffer.data(), tile_width, tile_height, GDT_Float32, 0, 0);

    if (err != CE_None)
    {
        throw std::runtime_error("Failed to write DSM tile to GeoTIFF");
    }
}

void generateDSMGeoTIFF(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                        const GeoCoord &coord_system, const std::string &output_path, int tile_size)
{
    spdlog::info("Generating DSM GeoTIFF: {}", output_path);

    PerformanceMeasure p("Generate DSM GeoTIFF");

    OrthoMosaicContext context = prepareOrthoMosaicContext(surfaces, graph, ImageResolution::FullResolution);

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
    auto last_log_time = start_time;

    for (int tile_y = 0; tile_y < num_tiles_y; tile_y++)
    {
        for (int tile_x = 0; tile_x < num_tiles_x; tile_x++)
        {
            processDSMTile(dataset.get(), tile_x, tile_y, tile_size, context.bounds, context.gsd, width, height,
                           surfaces, context.mean_camera_z);

            completed_tiles++;

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() >= 5 ||
                completed_tiles == total_tiles)
            {
                double progress = 100.0 * completed_tiles / total_tiles;
                spdlog::info("DSM Progress: {:.1f}% ({}/{} tiles, {} seconds)", progress, completed_tiles, total_tiles,
                             elapsed);
                last_log_time = now;
            }
        }
    }

    // Build internal overviews (pyramids) for faster display in GIS software
    spdlog::info("Building DSM overviews...");
    std::vector<int> overview_levels;
    int min_dim = std::min(width, height);
    for (int level = 2; level < min_dim; level *= 2)
    {
        overview_levels.push_back(level);
    }
    if (!overview_levels.empty())
    {
        CPLErr err = dataset->BuildOverviews("AVERAGE", static_cast<int>(overview_levels.size()),
                                             overview_levels.data(), 0, nullptr, nullptr, nullptr);
        if (err != CE_None)
        {
            spdlog::warn("Failed to build DSM overviews");
        }
    }

    spdlog::info("DSM GeoTIFF generation complete: {}", output_path);
}

void generateGeoTIFFOrthomosaic(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                                const GeoCoord &coord_system, const std::string &output_path, int tile_size)
{
    spdlog::info("Generating full-resolution GeoTIFF orthomosaic: {}", output_path);

    PerformanceMeasure p("Generate GeoTIFF orthomosaic");

    OrthoMosaicContext context = prepareOrthoMosaicContext(surfaces, graph, ImageResolution::FullResolution);

    spdlog::info("x range [{}; {}]  y range [{}; {}]  mean surface {}", context.bounds.min_x, context.bounds.max_x,
                 context.bounds.min_y, context.bounds.max_y, context.bounds.mean_surface_z);

    int width = static_cast<int>((context.bounds.max_x - context.bounds.min_x) / context.gsd);
    int height = static_cast<int>((context.bounds.max_y - context.bounds.min_y) / context.gsd);

    if (width <= 0)
        width = 100;
    if (height <= 0)
        height = 100;

    spdlog::info("GSD: {}  Output dimensions: {}x{} pixels", context.gsd, width, height);

    std::string wkt = coord_system.getWKT();
    GDALDatasetPtr dataset =
        createGeoTIFF(output_path, width, height, context.bounds.min_x, context.bounds.max_y, context.gsd, wkt);

    // Build rotation cache once for all tiles
    std::unordered_map<size_t, Eigen::Matrix3d> inv_rotation_cache;
    for (size_t node_id : context.involved_nodes)
    {
        const auto *node = graph.getNode(node_id);
        if (node)
        {
            inv_rotation_cache[node_id] = node->payload.orientation.inverse().toRotationMatrix();
        }
    }

    int num_tiles_x = (width + tile_size - 1) / tile_size;
    int num_tiles_y = (height + tile_size - 1) / tile_size;
    int total_tiles = num_tiles_x * num_tiles_y;

    spdlog::info("Processing {} tiles ({}x{} grid, tile size: {}x{})", total_tiles, num_tiles_x, num_tiles_y, tile_size,
                 tile_size);

    int completed_tiles = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_log_time = start_time;

    FullResolutionImageCache image_cache(10);

    for (int tile_y = 0; tile_y < num_tiles_y; tile_y++)
    {
        for (int tile_x = 0; tile_x < num_tiles_x; tile_x++)
        {
            processTile(dataset.get(), tile_x, tile_y, tile_size, context.bounds, context.gsd, width, height, surfaces,
                        graph, context.imageGPSLocations, context.average_camera_elevation, context.mean_camera_z,
                        inv_rotation_cache, image_cache);

            completed_tiles++;

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() >= 5 ||
                completed_tiles == total_tiles)
            {
                double progress = 100.0 * completed_tiles / total_tiles;
                spdlog::info("Progress: {:.1f}% ({}/{} tiles, {} seconds)", progress, completed_tiles, total_tiles,
                             elapsed);
                last_log_time = now;
            }
        }
    }

    // Build internal overviews (pyramids) for faster display in GIS software
    spdlog::info("Building overviews...");
    std::vector<int> overview_levels;
    int min_dim = std::min(width, height);
    for (int level = 2; level < min_dim; level *= 2)
    {
        overview_levels.push_back(level);
    }
    if (!overview_levels.empty())
    {
        CPLErr err = dataset->BuildOverviews("AVERAGE", static_cast<int>(overview_levels.size()),
                                             overview_levels.data(), 0, nullptr, nullptr, nullptr);
        if (err != CE_None)
        {
            spdlog::warn("Failed to build overviews");
        }
    }

    spdlog::info("GeoTIFF orthomosaic generation complete: {}", output_path);
}

} // namespace opencalibration::orthomosaic
