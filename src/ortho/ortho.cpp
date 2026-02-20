#include <opencalibration/ortho/ortho.hpp>

#include <opencalibration/ortho/blending.hpp>
#include <opencalibration/ortho/color_balance.hpp>

#include <ceres/jet.h>
#include <cpl_string.h>
#include <eigen3/Eigen/Eigenvalues>
#include <jk/KDTree.h>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/geometry/utils.hpp>
#include <opencalibration/ortho/gdal_dataset.hpp>
#include <opencalibration/ortho/image_cache.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/surface/intersect.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <mutex>
#include <omp.h>

namespace
{

// Convert (x,y) to Hilbert curve distance for a square of side 2^order
uint32_t xy2d(int order, int x, int y)
{
    uint32_t d = 0;
    for (int s = order / 2; s > 0; s /= 2)
    {
        int rx = (x & s) > 0 ? 1 : 0;
        int ry = (y & s) > 0 ? 1 : 0;
        d += s * s * ((3 * rx) ^ ry);
        // Rotate quadrant
        if (ry == 0)
        {
            if (rx == 1)
            {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::swap(x, y);
        }
    }
    return d;
}

// Generate tile indices sorted by Hilbert curve distance for better spatial locality
std::vector<std::pair<int, int>> hilbertTileOrder(int num_tiles_x, int num_tiles_y)
{
    // Find the smallest power of 2 that covers both dimensions
    int max_dim = std::max(num_tiles_x, num_tiles_y);
    int order = 1;
    while (order < max_dim)
        order *= 2;

    std::vector<std::pair<uint32_t, std::pair<int, int>>> tiles;
    tiles.reserve(num_tiles_x * num_tiles_y);
    for (int ty = 0; ty < num_tiles_y; ty++)
    {
        for (int tx = 0; tx < num_tiles_x; tx++)
        {
            tiles.push_back({xy2d(order, tx, ty), {tx, ty}});
        }
    }
    std::sort(tiles.begin(), tiles.end());

    std::vector<std::pair<int, int>> result;
    result.reserve(tiles.size());
    for (auto &t : tiles)
    {
        result.push_back(t.second);
    }
    return result;
}

Eigen::Vector2i size(const opencalibration::GenericRaster &raster)
{
    const auto getSize = [](const auto &rasterInstance) -> Eigen::Vector2i {
        return {rasterInstance.layers[0].pixels.rows(), rasterInstance.layers[0].pixels.cols()};
    };
    return std::visit(getSize, raster);
}

float normalizedImageRadius(double pixel_x, double pixel_y, int width, int height)
{
    if (width <= 0 || height <= 0)
        return 0.0f;

    double half_w = width * 0.5;
    double half_h = height * 0.5;
    double dx = (pixel_x - half_w) / half_w;
    double dy = (pixel_y - half_h) / half_h;

    // Scale by sqrt(2) so image corners map to radius 1.0, then clamp.
    constexpr double INV_SQRT_TWO = 0.7071067811865475;
    double radius = std::sqrt(dx * dx + dy * dy) * INV_SQRT_TWO;
    return static_cast<float>(std::clamp(radius, 0.0, 1.0));
}

std::pair<float, float> normalizedImagePosition(double pixel_x, double pixel_y, int width, int height)
{
    if (width <= 0 || height <= 0)
        return {0.0f, 0.0f};

    float nx = static_cast<float>((pixel_x - width * 0.5) / (width * 0.5));
    float ny = static_cast<float>((pixel_y - height * 0.5) / (height * 0.5));
    return {std::clamp(nx, -1.0f, 1.0f), std::clamp(ny, -1.0f, 1.0f)};
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
                            double output_gsd, const Eigen::Vector2d &pixel, cv::Vec3b &result)
    {
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
                    const cv::Vec3b &bgr = bgr_image.at<cv::Vec3b>(py, px);

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

// Clamp output resolution to not exceed sum of input image pixels
void clampOutputResolution(double &gsd, int &width, int &height, const OrthoMosaicContext &context,
                           const MeasurementGraph &graph, const char *stage_name = "")
{
    uint64_t total_input_pixels = 0;
    for (size_t node_id : context.involved_nodes)
    {
        const auto *node = graph.getNode(node_id);
        if (node)
        {
            const auto &img = node->payload;
            total_input_pixels += static_cast<uint64_t>(img.metadata.camera_info.width_px) *
                                  static_cast<uint64_t>(img.metadata.camera_info.height_px);
        }
    }

    uint64_t output_pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    if (output_pixels > total_input_pixels && total_input_pixels > 0)
    {
        // Scale GSD up to keep output resolution <= input pixels
        double scale_factor = std::sqrt(static_cast<double>(output_pixels) / total_input_pixels);
        gsd *= scale_factor;
        width = static_cast<int>(width / scale_factor);
        height = static_cast<int>(height / scale_factor);

        std::string stage_str = (stage_name && *stage_name) ? std::string(stage_name) + ": " : std::string("");
        spdlog::info("{}Clamped output resolution: GSD adjusted to {} (output pixels {} > input pixels {})", stage_str,
                     gsd, output_pixels, total_input_pixels);
    }
}

void clampOutputMegapixels(double &gsd, int &width, int &height, double max_output_megapixels,
                           const char *stage_name = "")
{
    if (!std::isfinite(max_output_megapixels) || max_output_megapixels <= 0.0)
    {
        return;
    }

    uint64_t output_pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    uint64_t max_output_pixels = static_cast<uint64_t>(max_output_megapixels * 1000000.0);
    if (max_output_pixels == 0 || output_pixels <= max_output_pixels)
    {
        return;
    }

    double scale_factor = std::sqrt(static_cast<double>(output_pixels) / static_cast<double>(max_output_pixels));
    gsd *= scale_factor;
    width = std::max(1, static_cast<int>(width / scale_factor));
    height = std::max(1, static_cast<int>(height / scale_factor));

    std::string stage_str = (stage_name && *stage_name) ? std::string(stage_name) + ": " : std::string("");
    spdlog::info("{}Applied max output megapixels {} MP: GSD adjusted to {} (output pixels {} > max {})", stage_str,
                 max_output_megapixels, gsd, output_pixels, max_output_pixels);
}

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

double calculateGSD(const MeasurementGraph &graph, const ankerl::unordered_dense::set<size_t> &involved_nodes,
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
        context.imageGPSLocations.addPoint({node->payload.position.x(), node->payload.position.y()}, node_id, false);
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

    int width = static_cast<int>(image_width);
    int height = static_cast<int>(image_height);
    clampOutputResolution(context.gsd, width, height, context, graph, "Thumbnail");
    image_width = width;
    image_height = height;

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
    ankerl::unordered_dense::map<size_t, CameraCache> camera_cache;
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

                auto closest5 = context.imageGPSLocations.searchKnn({x, y}, 5);

                // get image vertically closest
                for (const auto &closest : closest5)
                {
                    const auto *closestNode = graph.getNode(closest.payload);
                    const auto &payload = closestNode->payload;
                    const auto &cc = camera_cache.at(closest.payload);

                    Eigen::Vector3d camera_ray = cc.inv_rotation * (sample_point - payload.position);
                    if (camera_ray.z() <= 0)
                        continue;

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
}

GDALDatasetPtr createGeoTIFF(const std::string &path, int width, int height, double min_x, double max_y, double gsd,
                             const std::string &wkt)
{
    GDALAllRegister();

    GDALDriverH driver = GDALGetDriverByName("GTiff");
    if (!driver)
    {
        throw std::runtime_error("GTiff driver not available");
    }

    char **options = nullptr;
    options = CSLSetNameValue(options, "TILED", "YES");
    options = CSLSetNameValue(options, "BLOCKXSIZE", "512");
    options = CSLSetNameValue(options, "BLOCKYSIZE", "512");
    options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
    options = CSLSetNameValue(options, "PREDICTOR", "2");
    options = CSLSetNameValue(options, "PHOTOMETRIC", "RGB");
    options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");
    options = CSLSetNameValue(options, "SPARSE_OK", "YES");

    GDALDatasetH dataset = GDALCreate(driver, path.c_str(), width, height, 4, GDT_Byte, options);

    CSLDestroy(options);

    if (!dataset)
    {
        throw std::runtime_error("Failed to create GeoTIFF: " + path);
    }

    GDALDatasetWrapper ds_wrapper(dataset);

    // Set geotransform: [min_x, gsd, 0, max_y, 0, -gsd]
    double geotransform[6] = {min_x, gsd, 0, max_y, 0, -gsd};
    ds_wrapper.SetGeoTransform(geotransform);

    // Set projection
    if (!wkt.empty())
    {
        ds_wrapper.SetProjection(wkt.c_str());
    }

    // Set band interpretation
    GDALRasterBandWrapper band1(ds_wrapper.GetRasterBand(1));
    GDALRasterBandWrapper band2(ds_wrapper.GetRasterBand(2));
    GDALRasterBandWrapper band3(ds_wrapper.GetRasterBand(3));
    GDALRasterBandWrapper band4(ds_wrapper.GetRasterBand(4));
    band1.SetColorInterpretation(GCI_RedBand);
    band2.SetColorInterpretation(GCI_GreenBand);
    band3.SetColorInterpretation(GCI_BlueBand);
    band4.SetColorInterpretation(GCI_AlphaBand);

    return GDALDatasetPtr(dataset);
}

// Helper: Write tile data to GeoTIFF
void writeTileToGeoTIFF(GDALDatasetH dataset, int x_offset, int y_offset, int width, int height,
                        const std::vector<uint8_t> &buffer)
{
    // Deinterleave RGBA -> separate bands
    std::vector<uint8_t> band_data[4];
    for (auto &bd : band_data)
    {
        bd.resize(width * height);
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
        GDALRasterBandH hBand = GDALGetRasterBand(dataset, band);
        GDALRasterBandWrapper band_wrapper(hBand);
        CPLErr err = band_wrapper.RasterIO(GF_Write, x_offset, y_offset, width, height, band_data[band - 1].data(),
                                           width, height, GDT_Byte, 0, 0);

        if (err != CE_None)
        {
            throw std::runtime_error("Failed to write tile to GeoTIFF");
        }
    }
}

// Helper: Create GDAL GeoTIFF dataset for DSM (single float32 band)
GDALDatasetPtr createDSMGeoTIFF(const std::string &path, int width, int height, double min_x, double max_y, double gsd,
                                const std::string &wkt)
{
    GDALAllRegister();

    GDALDriverH driver = GDALGetDriverByName("GTiff");
    if (!driver)
    {
        throw std::runtime_error("GTiff driver not available");
    }

    char **options = nullptr;
    options = CSLSetNameValue(options, "TILED", "YES");
    options = CSLSetNameValue(options, "BLOCKXSIZE", "512");
    options = CSLSetNameValue(options, "BLOCKYSIZE", "512");
    options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
    options = CSLSetNameValue(options, "PREDICTOR", "2");
    options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");
    options = CSLSetNameValue(options, "SPARSE_OK", "YES");

    GDALDatasetH dataset = GDALCreate(driver, path.c_str(), width, height, 1, GDT_Float32, options);

    CSLDestroy(options);

    if (!dataset)
    {
        throw std::runtime_error("Failed to create DSM GeoTIFF: " + path);
    }

    GDALDatasetWrapper ds_wrapper(dataset);

    // Set geotransform: [min_x, gsd, 0, max_y, 0, -gsd]
    double geotransform[6] = {min_x, gsd, 0, max_y, 0, -gsd};
    ds_wrapper.SetGeoTransform(geotransform);

    // Set projection
    if (!wkt.empty())
    {
        ds_wrapper.SetProjection(wkt.c_str());
    }

    // Set nodata value for DSM
    GDALRasterBandWrapper band_wrapper(ds_wrapper.GetRasterBand(1));
    band_wrapper.SetNoDataValue(std::numeric_limits<float>::quiet_NaN());

    return GDALDatasetPtr(dataset);
}

std::vector<float> computeDSMTile(int tile_x, int tile_y, int tile_size, const OrthoMosaicBounds &bounds, double gsd,
                                  int output_width, int output_height, const std::vector<surface_model> &surfaces,
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

    return tile_buffer;
}

void writeDSMTile(GDALDatasetH dataset, int tile_x, int tile_y, int tile_size, int output_width, int output_height,
                  const std::vector<float> &tile_buffer)
{
    int x_offset = tile_x * tile_size;
    int y_offset = tile_y * tile_size;
    int tile_width = std::min(tile_size, output_width - x_offset);
    int tile_height = std::min(tile_size, output_height - y_offset);

    GDALRasterBandWrapper band_wrapper(GDALGetRasterBand(dataset, 1));
    CPLErr err =
        band_wrapper.RasterIO(GF_Write, x_offset, y_offset, tile_width, tile_height,
                              const_cast<float *>(tile_buffer.data()), tile_width, tile_height, GDT_Float32, 0, 0);

    if (err != CE_None)
    {
        throw std::runtime_error("Failed to write DSM tile to GeoTIFF");
    }
}

void generateDSMGeoTIFF(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                        const GeoCoord &coord_system, const std::string &output_path, int tile_size,
                        double max_output_megapixels)
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

    clampOutputResolution(context.gsd, width, height, context, graph, "DSM");
    clampOutputMegapixels(context.gsd, width, height, max_output_megapixels, "DSM");

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
            auto dsm_tile = computeDSMTile(tile_x, tile_y, tile_size, context.bounds, context.gsd, width, height,
                                           surfaces, context.mean_camera_z);
            writeDSMTile(dataset.get(), tile_x, tile_y, tile_size, width, height, dsm_tile);

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
        GDALDatasetWrapper ds_wrapper(dataset.get());
        CPLErr err =
            ds_wrapper.BuildOverviews("AVERAGE", static_cast<int>(overview_levels.size()), overview_levels.data());
        if (err != CE_None)
        {
            spdlog::warn("Failed to build DSM overviews");
        }
    }

    spdlog::info("DSM GeoTIFF generation complete: {}", output_path);
}

namespace
{

GDALDatasetPtr createMultiBandGeoTIFF(const std::string &path, int width, int height, double min_x, double max_y,
                                      double gsd, const std::string &wkt, int num_bands, GDALDataType data_type)
{
    GDALAllRegister();

    GDALDriverH driver = GDALGetDriverByName("GTiff");
    if (!driver)
    {
        throw std::runtime_error("GTiff driver not available");
    }

    char **options = nullptr;
    options = CSLSetNameValue(options, "TILED", "YES");
    options = CSLSetNameValue(options, "BLOCKXSIZE", "512");
    options = CSLSetNameValue(options, "BLOCKYSIZE", "512");
    options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
    options = CSLSetNameValue(options, "PREDICTOR", "2");
    options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");
    options = CSLSetNameValue(options, "SPARSE_OK", "YES");

    GDALDatasetH dataset = GDALCreate(driver, path.c_str(), width, height, num_bands, data_type, options);
    CSLDestroy(options);

    if (!dataset)
    {
        throw std::runtime_error("Failed to create multi-band GeoTIFF: " + path);
    }

    GDALDatasetWrapper ds_wrapper(dataset);

    double geotransform[6] = {min_x, gsd, 0, max_y, 0, -gsd};
    ds_wrapper.SetGeoTransform(geotransform);

    if (!wkt.empty())
    {
        ds_wrapper.SetProjection(wkt.c_str());
    }

    return GDALDatasetPtr(dataset);
}

ankerl::unordered_dense::set<size_t> findTileCameras(int tile_x, int tile_y, int tile_size,
                                                     const opencalibration::orthomosaic::OrthoMosaicBounds &bounds,
                                                     double gsd, int output_width, int output_height,
                                                     const jk::tree::KDTree<size_t, 2> &imageGPSLocations)
{
    PerformanceMeasure thread_perf("Ortho Stage 1 - read");

    int x_offset = tile_x * tile_size;
    int y_offset = tile_y * tile_size;
    int tile_width = std::min(tile_size, output_width - x_offset);
    int tile_height = std::min(tile_size, output_height - y_offset);

    ankerl::unordered_dense::set<size_t> camera_ids;
    jk::tree::KDTree<size_t, 2>::Searcher searcher(imageGPSLocations);

    int N = 10;
    for (int sy = 0; sy < N; sy++)
    {
        for (int sx = 0; sx < N; sx++)
        {
            int local_col = tile_width * sx / (N - 1);
            int local_row = tile_height * sy / (N - 1);
            local_col = std::min(local_col, tile_width);
            local_row = std::min(local_row, tile_height);

            int global_col = x_offset + local_col;
            int global_row = y_offset + local_row;

            double x = global_col * gsd + bounds.min_x;
            double y = bounds.max_y - global_row * gsd;

            const auto &closest5 = searcher.search({x, y}, INFINITY, 5);
            for (const auto &closest : closest5)
            {
                camera_ids.insert(closest.payload);
            }
        }
    }

    return camera_ids;
}

void prefetchImages(const ankerl::unordered_dense::set<size_t> &camera_ids,
                    const opencalibration::MeasurementGraph &graph,
                    opencalibration::orthomosaic::FullResolutionImageCache &image_cache)
{
    PerformanceMeasure thread_perf("Ortho Stage 1 - read");

    for (size_t cam_id : camera_ids)
    {
        const auto *node = graph.getNode(cam_id);
        if (node)
        {
            image_cache.getImage(cam_id, node->payload.path);
        }
    }
}

void writeLayeredTileToGeoTIFF(GDALDatasetH layers_ds, GDALDatasetH cameras_ds,
                               const opencalibration::orthomosaic::LayeredTileBuffer &tile, int x_offset, int y_offset)
{
    PerformanceMeasure thread_perf("Ortho Stage 1 - write");

    int w = tile.width;
    int h = tile.height;
    int N = tile.num_layers;

    for (int layer = 0; layer < N; layer++)
    {
        std::vector<uint8_t> band_b(w * h), band_g(w * h), band_r(w * h), band_a(w * h);
        std::vector<uint32_t> band_cam_lo(w * h, 0);
        std::vector<uint32_t> band_cam_hi(w * h, 0);

        for (int i = 0; i < w * h; i++)
        {
            const auto &sample = tile.layers[layer][i];
            if (sample.valid)
            {
                band_b[i] = sample.color_bgr[0];
                band_g[i] = sample.color_bgr[1];
                band_r[i] = sample.color_bgr[2];
                band_a[i] = 255;
                band_cam_lo[i] = static_cast<uint32_t>(sample.camera_id & 0xFFFFFFFF);
                band_cam_hi[i] = static_cast<uint32_t>((sample.camera_id >> 32) & 0xFFFFFFFF);
            }
            else
            {
                band_a[i] = 0;
            }
        }

        // Bands are: layer0_B, layer0_G, layer0_R, layer0_A, layer1_B, ...
        int band_offset = layer * 4;
        CPLErr err;
        GDALRasterBandWrapper band_b_wrapper(GDALGetRasterBand(layers_ds, band_offset + 1));
        err = band_b_wrapper.RasterIO(GF_Write, x_offset, y_offset, w, h, band_b.data(), w, h, GDT_Byte, 0, 0);
        if (err != CE_None)
            throw std::runtime_error("Failed to write layered tile band B");

        GDALRasterBandWrapper band_g_wrapper(GDALGetRasterBand(layers_ds, band_offset + 2));
        err = band_g_wrapper.RasterIO(GF_Write, x_offset, y_offset, w, h, band_g.data(), w, h, GDT_Byte, 0, 0);
        if (err != CE_None)
            throw std::runtime_error("Failed to write layered tile band G");

        GDALRasterBandWrapper band_r_wrapper(GDALGetRasterBand(layers_ds, band_offset + 3));
        err = band_r_wrapper.RasterIO(GF_Write, x_offset, y_offset, w, h, band_r.data(), w, h, GDT_Byte, 0, 0);
        if (err != CE_None)
            throw std::runtime_error("Failed to write layered tile band R");

        GDALRasterBandWrapper band_a_wrapper(GDALGetRasterBand(layers_ds, band_offset + 4));
        err = band_a_wrapper.RasterIO(GF_Write, x_offset, y_offset, w, h, band_a.data(), w, h, GDT_Byte, 0, 0);
        if (err != CE_None)
            throw std::runtime_error("Failed to write layered tile band A");

        // Camera ID bands (two uint32 bands per layer for full 64-bit size_t)
        int cam_band_offset = layer * 2;
        GDALRasterBandWrapper band_cam_lo_wrapper(GDALGetRasterBand(cameras_ds, cam_band_offset + 1));
        err = band_cam_lo_wrapper.RasterIO(GF_Write, x_offset, y_offset, w, h, band_cam_lo.data(), w, h, GDT_UInt32, 0,
                                           0);
        if (err != CE_None)
            throw std::runtime_error("Failed to write camera ID band (lo)");

        GDALRasterBandWrapper band_cam_hi_wrapper(GDALGetRasterBand(cameras_ds, cam_band_offset + 2));
        err = band_cam_hi_wrapper.RasterIO(GF_Write, x_offset, y_offset, w, h, band_cam_hi.data(), w, h, GDT_UInt32, 0,
                                           0);
        if (err != CE_None)
            throw std::runtime_error("Failed to write camera ID band (hi)");
    }
}

void readLayeredTileFromGeoTIFF(GDALDatasetH layers_ds, GDALDatasetH cameras_ds,
                                opencalibration::orthomosaic::LayeredTileBuffer &tile, int x_offset, int y_offset,
                                int w, int h, int num_layers)
{
    tile.resize(w, h, num_layers);

    for (int layer = 0; layer < num_layers; layer++)
    {
        std::vector<uint8_t> band_b(w * h), band_g(w * h), band_r(w * h), band_a(w * h);
        std::vector<uint32_t> band_cam_lo(w * h, 0);
        std::vector<uint32_t> band_cam_hi(w * h, 0);

        int band_offset = layer * 4;
        CPLErr err;
        GDALRasterBandWrapper band_b_wrapper(GDALGetRasterBand(layers_ds, band_offset + 1));
        err = band_b_wrapper.RasterIO(GF_Read, x_offset, y_offset, w, h, band_b.data(), w, h, GDT_Byte, 0, 0);
        if (err != CE_None)
            throw std::runtime_error("Failed to read layered tile band B");

        GDALRasterBandWrapper band_g_wrapper(GDALGetRasterBand(layers_ds, band_offset + 2));
        err = band_g_wrapper.RasterIO(GF_Read, x_offset, y_offset, w, h, band_g.data(), w, h, GDT_Byte, 0, 0);
        if (err != CE_None)
            throw std::runtime_error("Failed to read layered tile band G");

        GDALRasterBandWrapper band_r_wrapper(GDALGetRasterBand(layers_ds, band_offset + 3));
        err = band_r_wrapper.RasterIO(GF_Read, x_offset, y_offset, w, h, band_r.data(), w, h, GDT_Byte, 0, 0);
        if (err != CE_None)
            throw std::runtime_error("Failed to read layered tile band R");

        GDALRasterBandWrapper band_a_wrapper(GDALGetRasterBand(layers_ds, band_offset + 4));
        err = band_a_wrapper.RasterIO(GF_Read, x_offset, y_offset, w, h, band_a.data(), w, h, GDT_Byte, 0, 0);
        if (err != CE_None)
            throw std::runtime_error("Failed to read layered tile band A");

        int cam_band_offset = layer * 2;
        GDALRasterBandWrapper band_cam_lo_wrapper(GDALGetRasterBand(cameras_ds, cam_band_offset + 1));
        err =
            band_cam_lo_wrapper.RasterIO(GF_Read, x_offset, y_offset, w, h, band_cam_lo.data(), w, h, GDT_UInt32, 0, 0);
        if (err != CE_None)
            throw std::runtime_error("Failed to read camera ID band (lo)");

        GDALRasterBandWrapper band_cam_hi_wrapper(GDALGetRasterBand(cameras_ds, cam_band_offset + 2));
        err =
            band_cam_hi_wrapper.RasterIO(GF_Read, x_offset, y_offset, w, h, band_cam_hi.data(), w, h, GDT_UInt32, 0, 0);
        if (err != CE_None)
            throw std::runtime_error("Failed to read camera ID band (hi)");

        for (int i = 0; i < w * h; i++)
        {
            auto &sample = tile.layers[layer][i];
            if (band_a[i] > 0)
            {
                sample.color_bgr = cv::Vec3b(band_b[i], band_g[i], band_r[i]);
                sample.camera_id = static_cast<size_t>(band_cam_lo[i]) | (static_cast<size_t>(band_cam_hi[i]) << 32);
                sample.valid = true;
            }
            else
            {
                sample.valid = false;
            }
        }
    }
}

} // namespace

void processLayeredTile(int tile_x, int tile_y, int tile_size, const OrthoMosaicBounds &bounds, double gsd,
                        int output_width, int output_height, const std::vector<float> &dsm_tile,
                        const MeasurementGraph &graph, const jk::tree::KDTree<size_t, 2> &imageGPSLocations,
                        const ankerl::unordered_dense::map<size_t, Eigen::Matrix3d> &inv_rotation_cache,
                        FullResolutionImageCache &image_cache, int num_layers, LayeredTileBuffer &tile_out,
                        std::vector<ColorCorrespondence> &correspondences_out, int correspondence_subsample,
                        int correspondence_kernel_radius, std::mutex &correspondences_mutex)
{
    int x_offset = tile_x * tile_size;
    int y_offset = tile_y * tile_size;
    int tile_width = std::min(tile_size, output_width - x_offset);
    int tile_height = std::min(tile_size, output_height - y_offset);

    tile_out.resize(tile_width, tile_height, num_layers);

#pragma omp parallel
    {
        PerformanceMeasure thread_perf("Ortho Stage 1 - process");

        PatchSampler sampler;

        jk::tree::KDTree<size_t, 2>::Searcher tree_searcher(imageGPSLocations);
        ankerl::unordered_dense::map<uint64_t, cv::Mat> local_image_cache;

        std::vector<ColorCorrespondence> local_correspondences;

#pragma omp for schedule(dynamic)
        for (int local_row = 0; local_row < tile_height; local_row++)
        {
            for (int local_col = 0; local_col < tile_width; local_col++)
            {
                int global_col = x_offset + local_col;
                int global_row = y_offset + local_row;

                const double x = global_col * gsd + bounds.min_x;
                const double y = bounds.max_y - global_row * gsd;

                float z_f = dsm_tile[local_row * tile_width + local_col];
                if (std::isnan(z_f))
                    continue;
                double z = z_f;

                Eigen::Vector3d sample_point(x, y, z);
                const auto &closest5 = tree_searcher.search({x, y}, INFINITY, 5);

                int layer_idx = 0;
                for (const auto &closest : closest5)
                {
                    if (layer_idx >= num_layers)
                        break;

                    const auto *closestNode = graph.getNode(closest.payload);
                    const auto &payload = closestNode->payload;

                    auto inv_rot_it = inv_rotation_cache.find(closest.payload);
                    if (inv_rot_it == inv_rotation_cache.end())
                        continue;
                    const Eigen::Matrix3d &inv_rotation = inv_rot_it->second;

                    Eigen::Vector3d camera_ray = inv_rotation * (sample_point - payload.position);
                    if (camera_ray.z() <= 0)
                        continue;

                    Eigen::Vector2d pixel = image_from_3d(sample_point, *payload.model, payload.position, inv_rotation);

                    if (pixel.x() < 0 || pixel.x() >= payload.model->pixels_cols || pixel.y() < 0 ||
                        pixel.y() >= payload.model->pixels_rows)
                        continue;

                    cv::Mat full_image;
                    auto img_cache_iter = local_image_cache.find(closest.payload);
                    if (img_cache_iter != local_image_cache.end())
                    {
                        full_image = img_cache_iter->second;
                    }
                    else
                    {
                        full_image = image_cache.getImage(closest.payload, payload.path);
                        if (!full_image.empty())
                        {
                            local_image_cache.emplace(closest.payload, full_image);
                        }
                    }
                    if (full_image.empty())
                        continue;

                    cv::Vec3b color_bgr;
                    if (!sampler.sampleWithJacobian(full_image, sample_point, *payload.model, payload.position,
                                                    inv_rotation, gsd, pixel, color_bgr))
                        continue;

                    auto &sample = tile_out.at(layer_idx, local_row, local_col);
                    sample.color_bgr = color_bgr;
                    sample.camera_id = closest.payload;
                    sample.model_id = payload.model->id;
                    sample.valid = true;

                    sample.normalized_radius = normalizedImageRadius(pixel.x(), pixel.y(), payload.model->pixels_cols,
                                                                     payload.model->pixels_rows);

                    auto [nx, ny] = normalizedImagePosition(pixel.x(), pixel.y(), payload.model->pixels_cols,
                                                            payload.model->pixels_rows);
                    sample.normalized_x = nx;
                    sample.normalized_y = ny;

                    Eigen::Vector3d to_point = sample_point - payload.position;
                    Eigen::Vector3d camera_down = payload.orientation.inverse() * Eigen::Vector3d(0, 0, 1);
                    double cos_angle = camera_down.dot(to_point.normalized());
                    sample.view_angle = static_cast<float>(std::acos(std::clamp(cos_angle, -1.0, 1.0)));

                    float camera_dist = static_cast<float>(to_point.norm());
                    sample.weight =
                        computeBlendWeight(static_cast<float>(pixel.x()), static_cast<float>(pixel.y()),
                                           payload.model->pixels_cols, payload.model->pixels_rows, camera_dist);

                    layer_idx++;
                }

                // Voronoi boundary detection: check if rank-1 camera differs from neighbors
                if (layer_idx > 0 && correspondence_subsample > 0)
                {
                    size_t my_cam = tile_out.at(0, local_row, local_col).camera_id;
                    bool is_boundary = false;

                    const int dx_arr[] = {-1, 1, 0, 0};
                    const int dy_arr[] = {0, 0, -1, 1};
                    for (int d = 0; d < 4 && !is_boundary; d++)
                    {
                        int nr = local_row + dy_arr[d];
                        int nc = local_col + dx_arr[d];
                        if (nr >= 0 && nr < tile_height && nc >= 0 && nc < tile_width)
                        {
                            const auto &neighbor = tile_out.at(0, nr, nc);
                            if (neighbor.valid && neighbor.camera_id != my_cam)
                            {
                                is_boundary = true;
                            }
                        }
                    }

                    bool sample_this_pixel = false;
                    if (is_boundary && ((local_row + local_col) % correspondence_subsample == 0))
                        sample_this_pixel = true;
                    else if (!is_boundary && (local_row % correspondence_subsample == 0) &&
                             (local_col % correspondence_subsample == 0))
                        sample_this_pixel = true;

                    if (sample_this_pixel)
                    {
                        for (int a = 0; a < layer_idx; a++)
                        {
                            for (int b = a + 1; b < layer_idx; b++)
                            {
                                const auto &sa = tile_out.at(a, local_row, local_col);
                                const auto &sb = tile_out.at(b, local_row, local_col);

                                cv::Vec3f sum_lab_a(0, 0, 0), sum_lab_b(0, 0, 0);
                                int count = 0;
                                for (int dr = -correspondence_kernel_radius; dr <= correspondence_kernel_radius; dr++)
                                {
                                    for (int dc = -correspondence_kernel_radius; dc <= correspondence_kernel_radius;
                                         dc++)
                                    {
                                        int kr = local_row + dr;
                                        int kc = local_col + dc;
                                        if (kr < 0 || kr >= tile_height || kc < 0 || kc >= tile_width)
                                            continue;
                                        const auto &ka = tile_out.at(a, kr, kc);
                                        const auto &kb = tile_out.at(b, kr, kc);
                                        if (!ka.valid || !kb.valid || ka.camera_id != sa.camera_id ||
                                            kb.camera_id != sb.camera_id)
                                            continue;

                                        cv::Mat bgr_a(1, 1, CV_8UC3), bgr_b(1, 1, CV_8UC3);
                                        bgr_a.at<cv::Vec3b>(0, 0) = ka.color_bgr;
                                        bgr_b.at<cv::Vec3b>(0, 0) = kb.color_bgr;
                                        cv::Mat bgr_a_f, bgr_b_f, lab_a, lab_b;
                                        bgr_a.convertTo(bgr_a_f, CV_32FC3, 1.0 / 255.0);
                                        bgr_b.convertTo(bgr_b_f, CV_32FC3, 1.0 / 255.0);
                                        cv::cvtColor(bgr_a_f, lab_a, cv::COLOR_BGR2Lab);
                                        cv::cvtColor(bgr_b_f, lab_b, cv::COLOR_BGR2Lab);
                                        sum_lab_a += lab_a.at<cv::Vec3f>(0, 0);
                                        sum_lab_b += lab_b.at<cv::Vec3f>(0, 0);
                                        count++;
                                    }
                                }
                                if (count == 0)
                                    continue;

                                cv::Vec3f avg_lab_a = sum_lab_a / static_cast<float>(count);
                                cv::Vec3f avg_lab_b = sum_lab_b / static_cast<float>(count);

                                ColorCorrespondence corr;
                                corr.lab_a = {avg_lab_a[0], avg_lab_a[1], avg_lab_a[2]};
                                corr.lab_b = {avg_lab_b[0], avg_lab_b[1], avg_lab_b[2]};
                                corr.camera_id_a = sa.camera_id;
                                corr.camera_id_b = sb.camera_id;
                                corr.model_id_a = sa.model_id;
                                corr.model_id_b = sb.model_id;
                                corr.normalized_radius_a = sa.normalized_radius;
                                corr.normalized_radius_b = sb.normalized_radius;
                                corr.view_angle_a = sa.view_angle;
                                corr.view_angle_b = sb.view_angle;
                                corr.normalized_x_a = sa.normalized_x;
                                corr.normalized_y_a = sa.normalized_y;
                                corr.normalized_x_b = sb.normalized_x;
                                corr.normalized_y_b = sb.normalized_y;

                                local_correspondences.push_back(corr);
                            }
                        }
                    }
                }
            }
        }

        if (!local_correspondences.empty())
        {
            std::lock_guard<std::mutex> lock(correspondences_mutex);
            correspondences_out.insert(correspondences_out.end(), local_correspondences.begin(),
                                       local_correspondences.end());
        }
    }
}

std::vector<ColorCorrespondence> generateLayeredGeoTIFF(const std::vector<surface_model> &surfaces,
                                                        const MeasurementGraph &graph, const GeoCoord &coord_system,
                                                        const std::string &layers_path, const std::string &cameras_path,
                                                        const std::string &dsm_output_path,
                                                        const OrthoMosaicConfig &config)
{
    spdlog::info("Pass 1: Generating layered GeoTIFF: {}", layers_path);
    PerformanceMeasure p("Ortho stage 1 - setup");

    GDALAllRegister();

    OrthoMosaicContext context = prepareOrthoMosaicContext(surfaces, graph, ImageResolution::FullResolution);

    int width = static_cast<int>((context.bounds.max_x - context.bounds.min_x) / context.gsd);
    int height = static_cast<int>((context.bounds.max_y - context.bounds.min_y) / context.gsd);

    if (width <= 0)
        width = 100;
    if (height <= 0)
        height = 100;

    clampOutputResolution(context.gsd, width, height, context, graph, "Layered");
    clampOutputMegapixels(context.gsd, width, height, config.max_output_megapixels, "Layered");

    const OrthoMosaicBounds &bounds = context.bounds;
    double gsd = context.gsd;

    spdlog::info("GSD: {}  Output dimensions: {}x{} pixels, {} layers", gsd, width, height, config.num_layers);

    std::string wkt = coord_system.getWKT();

    // Create intermediate multi-band GeoTIFF: N * 4 bands (BGRA per layer)
    int num_color_bands = config.num_layers * 4;
    GDALDatasetPtr layers_ds = createMultiBandGeoTIFF(layers_path, width, height, bounds.min_x, bounds.max_y, gsd, wkt,
                                                      num_color_bands, GDT_Byte);

    // Create sidecar camera ID GeoTIFF: N bands of uint32
    GDALDatasetPtr cameras_ds = createMultiBandGeoTIFF(cameras_path, width, height, bounds.min_x, bounds.max_y, gsd,
                                                       wkt, config.num_layers * 2, GDT_UInt32);

    // Create DSM GeoTIFF alongside layers
    GDALDatasetPtr dsm_ds = createDSMGeoTIFF(dsm_output_path, width, height, bounds.min_x, bounds.max_y, gsd, wkt);

    ankerl::unordered_dense::map<size_t, Eigen::Matrix3d> inv_rotation_cache;
    for (size_t node_id : context.involved_nodes)
    {
        const auto *node = graph.getNode(node_id);
        if (node)
        {
            inv_rotation_cache[node_id] = node->payload.orientation.inverse().toRotationMatrix();
        }
    }

    int tile_size = config.tile_size;
    int num_tiles_x = (width + tile_size - 1) / tile_size;
    int num_tiles_y = (height + tile_size - 1) / tile_size;
    int total_tiles = num_tiles_x * num_tiles_y;

    spdlog::info("Pass 1: Processing {} tiles ({}x{} grid)", total_tiles, num_tiles_x, num_tiles_y);

    std::vector<ColorCorrespondence> all_correspondences;
    std::mutex correspondences_mutex;

    int completed_tiles = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_log_time = start_time;

    p.reset("");

    FullResolutionImageCache image_cache(config.num_layers * 10);

    auto tile_order = hilbertTileOrder(num_tiles_x, num_tiles_y);

    // Prefetch images for the first tile synchronously
    if (!tile_order.empty())
    {
        auto first_cams = findTileCameras(tile_order[0].first, tile_order[0].second, tile_size, bounds, gsd, width,
                                          height, context.imageGPSLocations);
        prefetchImages(first_cams, graph, image_cache);
    }

    std::future<void> write_future;
    std::future<void> prefetch_future;

    for (size_t i = 0; i < tile_order.size(); i++)
    {
        const auto &[tile_x, tile_y] = tile_order[i];

        if (i + 1 < tile_order.size())
        {
            if (prefetch_future.valid())
                prefetch_future.wait();
            const auto &next_tile = tile_order[i + 1];
            const auto next_tx = next_tile.first;
            const auto next_ty = next_tile.second;
            prefetch_future = std::async(std::launch::async, [&, next_tx, next_ty] {
                auto cams =
                    findTileCameras(next_tx, next_ty, tile_size, bounds, gsd, width, height, context.imageGPSLocations);
                prefetchImages(cams, graph, image_cache);
            });
        }

        std::vector<float> dsm_tile =
            computeDSMTile(tile_x, tile_y, tile_size, bounds, gsd, width, height, surfaces, context.mean_camera_z);

        LayeredTileBuffer tile_buf;
        processLayeredTile(tile_x, tile_y, tile_size, bounds, gsd, width, height, dsm_tile, graph,
                           context.imageGPSLocations, inv_rotation_cache, image_cache, config.num_layers, tile_buf,
                           all_correspondences, config.correspondence_subsample, config.correspondence_kernel_radius,
                           correspondences_mutex);

        if (write_future.valid())
            write_future.wait();

        int x_off = tile_x * tile_size;
        int y_off = tile_y * tile_size;
        auto tile_buf_ptr = std::make_shared<LayeredTileBuffer>(std::move(tile_buf));
        auto dsm_tile_ptr = std::make_shared<std::vector<float>>(std::move(dsm_tile));
        write_future = std::async(std::launch::async, [&, tile_buf_ptr, dsm_tile_ptr, x_off, y_off, tile_x, tile_y] {
            writeLayeredTileToGeoTIFF(layers_ds.get(), cameras_ds.get(), *tile_buf_ptr, x_off, y_off);
            writeDSMTile(dsm_ds.get(), tile_x, tile_y, tile_size, width, height, *dsm_tile_ptr);
        });

        completed_tiles++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() >= 5 ||
            completed_tiles == total_tiles)
        {
            double progress = 100.0 * completed_tiles / total_tiles;
            spdlog::info("Pass 1 Progress: {:.1f}% ({}/{} tiles, {} seconds, {} correspondences)", progress,
                         completed_tiles, total_tiles, elapsed, all_correspondences.size());
            last_log_time = now;
        }
    }

    if (write_future.valid())
        write_future.wait();

    // Build DSM overviews
    spdlog::info("Building DSM overviews...");
    std::vector<int> overview_levels;
    int min_dim = std::min(width, height);
    for (int level = 2; level < min_dim; level *= 2)
    {
        overview_levels.push_back(level);
    }
    if (!overview_levels.empty())
    {
        GDALDatasetWrapper dsm_wrapper(dsm_ds.get());
        CPLErr err =
            dsm_wrapper.BuildOverviews("AVERAGE", static_cast<int>(overview_levels.size()), overview_levels.data());
        if (err != CE_None)
        {
            spdlog::warn("Failed to build DSM overviews");
        }
    }

    spdlog::info("Pass 1 complete: {} correspondences collected", all_correspondences.size());
    return all_correspondences;
}

void blendLayeredGeoTIFF(const std::string &layers_path, const std::string &cameras_path, const std::string &dsm_path,
                         const std::string &output_path, const ColorBalanceResult &color_balance,
                         const MeasurementGraph &graph, const GeoCoord &coord_system, const OrthoMosaicConfig &config)
{
    spdlog::info("Pass 2: Blending layered GeoTIFF to: {}", output_path);

    PerformanceMeasure p("Ortho pre-stage 2");

    GDALAllRegister();
    GDALDatasetPtr layers_ds(GDALOpen(layers_path.c_str(), GA_ReadOnly));
    GDALDatasetPtr cameras_ds(GDALOpen(cameras_path.c_str(), GA_ReadOnly));

    if (!layers_ds || !cameras_ds)
    {
        throw std::runtime_error("Failed to open intermediate GeoTIFFs for reading");
    }

    GDALDatasetWrapper layers_wrapper(layers_ds.get());
    int width = layers_wrapper.GetRasterXSize();
    int height = layers_wrapper.GetRasterYSize();
    int num_color_bands = layers_wrapper.GetRasterCount();
    int num_layers = num_color_bands / 4;

    double geotransform[6];
    layers_wrapper.GetGeoTransform(geotransform);
    double min_x = geotransform[0];
    double max_y = geotransform[3];
    double gsd = geotransform[1];

    spdlog::info("Pass 2: {}x{} pixels, {} layers, GSD {}", width, height, num_layers, gsd);

    // Close shared read handles; each thread will open its own
    layers_ds.reset();
    cameras_ds.reset();

    std::string wkt = coord_system.getWKT();
    GDALDatasetPtr output_ds = createGeoTIFF(output_path, width, height, min_x, max_y, gsd, wkt);

    // Build inv_rotation_cache from graph
    std::unordered_map<size_t, Eigen::Matrix3d> inv_rotation_cache;
    for (auto iter = graph.cnodebegin(); iter != graph.cnodeend(); ++iter)
    {
        if (iter->second.payload.orientation.coeffs().allFinite())
        {
            inv_rotation_cache[iter->first] = iter->second.payload.orientation.inverse().toRotationMatrix();
        }
    }

    int tile_size = config.tile_size;
    int num_tiles_x = (width + tile_size - 1) / tile_size;
    int num_tiles_y = (height + tile_size - 1) / tile_size;
    int total_tiles = num_tiles_x * num_tiles_y;

    spdlog::info("Pass 2: Processing {} tiles", total_tiles);

    std::atomic<int> completed_tiles{0};
    auto start_time = std::chrono::steady_clock::now();
    std::atomic<long long> last_log_seconds{-5};

    p.reset("");
    std::mutex gdal_write_mutex;

#pragma omp parallel
    {
        PerformanceMeasure thread_perf("Ortho stage 2");

        // Each thread opens its own read-only handles to avoid serializing reads
        GDALDatasetPtr thread_layers_ds(GDALOpen(layers_path.c_str(), GA_ReadOnly));
        GDALDatasetPtr thread_cameras_ds(GDALOpen(cameras_path.c_str(), GA_ReadOnly));
        GDALDatasetPtr thread_dsm_ds(GDALOpen(dsm_path.c_str(), GA_ReadOnly));

#pragma omp for schedule(dynamic)
        for (int tile_idx = 0; tile_idx < total_tiles; tile_idx++)
        {
            int tile_y = tile_idx / num_tiles_x;
            int tile_x = tile_idx % num_tiles_x;

            int x_offset = tile_x * tile_size;
            int y_offset = tile_y * tile_size;
            int tw = std::min(tile_size, width - x_offset);
            int th = std::min(tile_size, height - y_offset);

            LayeredTileBuffer tile_buf;
            readLayeredTileFromGeoTIFF(thread_layers_ds.get(), thread_cameras_ds.get(), tile_buf, x_offset, y_offset,
                                       tw, th, num_layers);

            std::vector<float> dsm_tile(tw * th, NAN);
            if (thread_dsm_ds)
            {
                GDALRasterBandWrapper dsm_band(GDALGetRasterBand(thread_dsm_ds.get(), 1));
                dsm_band.RasterIO(GF_Read, x_offset, y_offset, tw, th, dsm_tile.data(), tw, th, GDT_Float32, 0, 0);
            }

            std::vector<cv::Mat> bgr_layers(num_layers);
            std::vector<cv::Mat> weight_maps(num_layers);
            for (int layer = 0; layer < num_layers; layer++)
            {
                bgr_layers[layer] = cv::Mat(th, tw, CV_8UC3, cv::Scalar(0, 0, 0));
                weight_maps[layer] = cv::Mat::zeros(th, tw, CV_32FC1);
            }

            for (int local_row = 0; local_row < th; local_row++)
            {
                for (int local_col = 0; local_col < tw; local_col++)
                {
                    int global_col = x_offset + local_col;
                    int global_row = y_offset + local_row;
                    const double wx = global_col * gsd + min_x;
                    const double wy = max_y - global_row * gsd;

                    float dsm_z = dsm_tile[local_row * tw + local_col];
                    if (std::isnan(dsm_z))
                        continue;

                    for (int layer = 0; layer < num_layers; layer++)
                    {
                        auto &sample = tile_buf.at(layer, local_row, local_col);
                        if (!sample.valid)
                            continue;

                        const auto *node = graph.getNode(sample.camera_id);
                        if (!node)
                            continue;
                        const auto &payload = node->payload;

                        auto inv_rot_it = inv_rotation_cache.find(sample.camera_id);
                        if (inv_rot_it == inv_rotation_cache.end())
                            continue;

                        Eigen::Vector3d world_pt(wx, wy, static_cast<double>(dsm_z));

                        Eigen::Vector3d camera_ray = inv_rot_it->second * (world_pt - payload.position);
                        if (camera_ray.z() <= 0)
                        {
                            sample.valid = false;
                            continue;
                        }

                        Eigen::Vector2d pixel =
                            image_from_3d(world_pt, *payload.model, payload.position, inv_rot_it->second);

                        sample.normalized_radius = normalizedImageRadius(
                            pixel.x(), pixel.y(), payload.model->pixels_cols, payload.model->pixels_rows);

                        auto [nx, ny] = normalizedImagePosition(pixel.x(), pixel.y(), payload.model->pixels_cols,
                                                                payload.model->pixels_rows);
                        sample.normalized_x = nx;
                        sample.normalized_y = ny;

                        Eigen::Vector3d to_point = world_pt - payload.position;
                        Eigen::Vector3d cam_down = payload.orientation.inverse() * Eigen::Vector3d(0, 0, 1);
                        double cos_angle = cam_down.dot(to_point.normalized());
                        sample.view_angle = static_cast<float>(std::acos(std::clamp(cos_angle, -1.0, 1.0)));

                        float camera_dist = static_cast<float>(to_point.norm());
                        sample.weight =
                            computeBlendWeight(static_cast<float>(pixel.x()), static_cast<float>(pixel.y()),
                                               payload.model->pixels_cols, payload.model->pixels_rows, camera_dist);

                        bgr_layers[layer].at<cv::Vec3b>(local_row, local_col) = sample.color_bgr;
                        weight_maps[layer].at<float>(local_row, local_col) = sample.weight;
                    }
                }
            }

            std::vector<cv::Mat> lab_layers(num_layers);
            for (int layer = 0; layer < num_layers; layer++)
            {
                cv::Mat bgr_float;
                bgr_layers[layer].convertTo(bgr_float, CV_32FC3, 1.0 / 255.0);
                cv::cvtColor(bgr_float, lab_layers[layer], cv::COLOR_BGR2Lab);
            }

            for (int local_row = 0; local_row < th; local_row++)
            {
                for (int local_col = 0; local_col < tw; local_col++)
                {
                    for (int layer = 0; layer < num_layers; layer++)
                    {
                        const auto &sample = tile_buf.at(layer, local_row, local_col);
                        if (!sample.valid)
                            continue;

                        auto img_it = color_balance.per_image_params.find(sample.camera_id);
                        if (img_it == color_balance.per_image_params.end())
                            continue;
                        const auto &img_params = img_it->second;

                        cv::Vec3f &lab = lab_layers[layer].at<cv::Vec3f>(local_row, local_col);
                        lab[0] -= static_cast<float>(img_params.lab_offset[0]);
                        lab[1] -= static_cast<float>(img_params.lab_offset[1]);
                        lab[2] -= static_cast<float>(img_params.lab_offset[2]);

                        auto mdl_it = color_balance.per_model_params.find(sample.model_id);
                        if (mdl_it != color_balance.per_model_params.end())
                        {
                            const auto &vig = mdl_it->second;
                            float r2 = sample.normalized_radius * sample.normalized_radius;
                            float vig_corr = static_cast<float>(vig.coeffs[0]) * r2 +
                                             static_cast<float>(vig.coeffs[1]) * r2 * r2 +
                                             static_cast<float>(vig.coeffs[2]) * r2 * r2 * r2;
                            lab[0] -= vig_corr;
                        }
                        lab[0] -= static_cast<float>(img_params.brdf_coeff) * sample.view_angle * sample.view_angle;
                        lab[0] -= static_cast<float>(img_params.slope[0]) * sample.normalized_x +
                                  static_cast<float>(img_params.slope[1]) * sample.normalized_y;

                        lab[0] = std::clamp(lab[0], 0.0f, 100.0f);
                        lab[1] = std::clamp(lab[1], -127.0f, 127.0f);
                        lab[2] = std::clamp(lab[2], -127.0f, 127.0f);
                    }
                }
            }

            cv::Mat boundary_mask(th, tw, CV_8UC1, cv::Scalar(0));
            const int dx_arr[] = {-1, 1, 0, 0};
            const int dy_arr[] = {0, 0, -1, 1};
            for (int local_row = 0; local_row < th; local_row++)
            {
                for (int local_col = 0; local_col < tw; local_col++)
                {
                    const auto &sample = tile_buf.at(0, local_row, local_col);
                    if (!sample.valid)
                        continue;
                    for (int d = 0; d < 4; d++)
                    {
                        int nr = local_row + dy_arr[d], nc = local_col + dx_arr[d];
                        if (nr >= 0 && nr < th && nc >= 0 && nc < tw)
                        {
                            const auto &neighbor = tile_buf.at(0, nr, nc);
                            if (!neighbor.valid || neighbor.camera_id != sample.camera_id)
                            {
                                boundary_mask.at<uint8_t>(local_row, local_col) = 255;
                                break;
                            }
                        }
                    }
                }
            }

            cv::Mat inv_boundary;
            cv::bitwise_not(boundary_mask, inv_boundary);
            cv::Mat boundary_dist;
            cv::distanceTransform(inv_boundary, boundary_dist, cv::DIST_L2, 3);

            float steepness = std::log(99.0f) / static_cast<float>(config.blend_transition_radius);
            for (int layer = 1; layer < num_layers; layer++)
            {
                for (int local_row = 0; local_row < th; local_row++)
                {
                    for (int local_col = 0; local_col < tw; local_col++)
                    {
                        float d = boundary_dist.at<float>(local_row, local_col);
                        float falloff = 1.0f / (1.0f + std::exp(steepness * d));
                        weight_maps[layer].at<float>(local_row, local_col) *= falloff;
                    }
                }
            }

            cv::Mat blended = laplacianBlend(lab_layers, weight_maps, config.pyramid_levels);

            if (!blended.empty())
            {
                // Convert BGRA to RGBA for the output GeoTIFF
                std::vector<uint8_t> rgba_buffer(tw * th * 4);
                for (int i = 0; i < tw * th; i++)
                {
                    int r = i / tw;
                    int c = i % tw;
                    cv::Vec4b pixel = blended.at<cv::Vec4b>(r, c);
                    rgba_buffer[i * 4 + 0] = pixel[2]; // R (from B in BGRA)
                    rgba_buffer[i * 4 + 1] = pixel[1]; // G
                    rgba_buffer[i * 4 + 2] = pixel[0]; // B (from R in BGRA)
                    rgba_buffer[i * 4 + 3] = pixel[3]; // A

                    bool any_valid = false;
                    for (int l = 0; l < num_layers && !any_valid; l++)
                    {
                        any_valid = tile_buf.at(l, r, c).valid;
                    }
                    if (!any_valid)
                    {
                        uint8_t grey = ((y_offset + r) + (x_offset + c)) % 2 == 0 ? 64 : 128;
                        rgba_buffer[i * 4 + 0] = grey;
                        rgba_buffer[i * 4 + 1] = grey;
                        rgba_buffer[i * 4 + 2] = grey;
                        rgba_buffer[i * 4 + 3] = 0;
                    }
                }

                {
                    std::lock_guard<std::mutex> lock(gdal_write_mutex);
                    writeTileToGeoTIFF(output_ds.get(), x_offset, y_offset, tw, th, rgba_buffer);
                }
            }

            int done = ++completed_tiles;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            long long prev_log = last_log_seconds.load(std::memory_order_relaxed);
            if (done == total_tiles ||
                (elapsed - prev_log >= 5 && last_log_seconds.compare_exchange_strong(prev_log, elapsed)))
            {
                double progress = 100.0 * done / total_tiles;
                spdlog::info("Pass 2 Progress: {:.1f}% ({}/{} tiles, {} seconds)", progress, done, total_tiles,
                             elapsed);
            }
        }
    } // omp parallel

    spdlog::info("Building overviews...");
    std::vector<int> overview_levels;
    int min_dim = std::min(width, height);
    for (int level = 2; level < min_dim; level *= 2)
    {
        overview_levels.push_back(level);
    }
    if (!overview_levels.empty())
    {
        GDALDatasetWrapper output_wrapper(output_ds.get());
        CPLErr err =
            output_wrapper.BuildOverviews("AVERAGE", static_cast<int>(overview_levels.size()), overview_levels.data());
        if (err != CE_None)
        {
            spdlog::warn("Failed to build overviews");
        }
    }

    VSIUnlink(layers_path.c_str());
    VSIUnlink(cameras_path.c_str());

    spdlog::info("Pass 2 complete: {}", output_path);
}

void generateTexturedOBJ(const std::vector<surface_model> &surfaces, const std::string &geotiff_path,
                         const std::string &obj_path)
{
    GDALAllRegister();
    GDALDatasetPtr dataset = openGDALDataset(geotiff_path);
    if (!dataset)
    {
        spdlog::error("Failed to open GeoTIFF: {}", geotiff_path);
        return;
    }

    GDALDatasetWrapper ds(dataset.get());
    int img_width = ds.GetRasterXSize();
    int img_height = ds.GetRasterYSize();

    double geotransform[6];
    if (ds.GetGeoTransform(geotransform) != CE_None)
    {
        spdlog::error("Failed to read geotransform from {}", geotiff_path);
        return;
    }

    double min_x = geotransform[0];
    double max_y = geotransform[3];
    double gsd_x = geotransform[1];
    double gsd_y = -geotransform[5]; // geotransform[5] is negative

    // Derive output paths from obj_path
    std::string base_path = obj_path;
    if (base_path.size() >= 4 && base_path.substr(base_path.size() - 4) == ".obj")
    {
        base_path = base_path.substr(0, base_path.size() - 4);
    }
    std::string mtl_path = base_path + ".mtl";
    std::string jpg_path = base_path + ".jpg";

    // Extract just filenames for references within OBJ/MTL
    auto filename_only = [](const std::string &path) {
        size_t pos = path.find_last_of("/\\");
        return (pos != std::string::npos) ? path.substr(pos + 1) : path;
    };
    std::string mtl_filename = filename_only(mtl_path);
    std::string jpg_filename = filename_only(jpg_path);

    // Read RGBA from GeoTIFF and write JPEG texture
    int num_bands = ds.GetRasterCount();
    int bands_to_read = std::min(num_bands, 3);

    cv::Mat texture(img_height, img_width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int b = 0; b < bands_to_read; b++)
    {
        std::vector<uint8_t> band_data(img_width * img_height);
        GDALRasterBandWrapper band(ds.GetRasterBand(b + 1));
        CPLErr err = band.RasterIO(GF_Read, 0, 0, img_width, img_height, band_data.data(), img_width, img_height,
                                   GDT_Byte, 0, 0);
        if (err != CE_None)
        {
            spdlog::error("Failed to read band {} from GeoTIFF", b + 1);
            return;
        }
        // GDAL bands are R=1, G=2, B=3; OpenCV channels are BGR
        int cv_channel = (b == 0) ? 2 : (b == 2) ? 0 : 1;
        for (int y = 0; y < img_height; y++)
        {
            for (int x = 0; x < img_width; x++)
            {
                texture.at<cv::Vec3b>(y, x)[cv_channel] = band_data[y * img_width + x];
            }
        }
    }
    cv::imwrite(jpg_path, texture);
    spdlog::info("Wrote texture: {} ({}x{})", jpg_path, img_width, img_height);

    // Write MTL file
    {
        std::ofstream mtl(mtl_path);
        if (!mtl.is_open())
        {
            spdlog::error("Failed to open MTL file for writing: {}", mtl_path);
            return;
        }
        mtl << "newmtl orthomosaic_material\n";
        mtl << "Ka 1.0 1.0 1.0\n";
        mtl << "Kd 1.0 1.0 1.0\n";
        mtl << "Ks 0.0 0.0 0.0\n";
        mtl << "map_Kd " << jpg_filename << "\n";
    }
    spdlog::info("Wrote material: {}", mtl_path);

    // Write OBJ file
    std::ofstream obj(obj_path);
    if (!obj.is_open())
    {
        spdlog::error("Failed to open OBJ file for writing: {}", obj_path);
        return;
    }

    obj << "mtllib " << mtl_filename << "\n";
    obj << "usemtl orthomosaic_material\n";

    double extent_x = img_width * gsd_x;
    double extent_y = img_height * gsd_y;

    size_t global_vertex_offset = 0;

    for (size_t si = 0; si < surfaces.size(); si++)
    {
        const auto &mesh = surfaces[si].mesh;
        if (mesh.size_edges() == 0)
            continue;

        // Collect and sort vertices
        std::vector<size_t> sorted_nodes;
        sorted_nodes.reserve(mesh.size_nodes());
        std::transform(mesh.cnodebegin(), mesh.cnodeend(), std::back_inserter(sorted_nodes),
                       [](const auto &iter) { return iter.first; });
        std::sort(sorted_nodes.begin(), sorted_nodes.end());

        // Map from node ID to sequential index (1-based for OBJ, offset by global count)
        std::unordered_map<size_t, size_t> node_to_index;

        // Write vertices and UVs
        for (size_t node_id : sorted_nodes)
        {
            const auto &loc = mesh.getNode(node_id)->payload.location;
            node_to_index[node_id] = global_vertex_offset + node_to_index.size() + 1; // 1-based OBJ indices

            obj << "v " << loc.x() << " " << loc.y() << " " << loc.z() << "\n";

            double u = (loc.x() - min_x) / extent_x;
            double v = 1.0 - (max_y - loc.y()) / extent_y;
            obj << "vt " << u << " " << v << "\n";
        }

        // Extract faces (same logic as PLY serializer)
        auto nodes_anticlockwise = [&mesh](const std::array<size_t, 3> &face) {
            std::array<Eigen::Vector3d, 3> corners;
            Eigen::Index i = 0;
            for (size_t node_id : face)
            {
                corners[i++] = mesh.getNode(node_id)->payload.location;
            }
            return anticlockwise(corners);
        };

        struct ArrayHash
        {
            size_t operator()(const std::array<size_t, 3> &arr) const
            {
                size_t result = arr.size();
                for (auto &i : arr)
                {
                    result ^= i + 0x9e3779b9 + (result << 6) + (result >> 2);
                }
                return result;
            }
        };

        std::unordered_set<std::array<size_t, 3>, ArrayHash> faces;
        std::vector<size_t> sorted_edges;
        sorted_edges.reserve(mesh.size_edges());
        std::transform(mesh.cedgebegin(), mesh.cedgeend(), std::back_inserter(sorted_edges),
                       [](const auto &iter) { return iter.first; });
        std::sort(sorted_edges.begin(), sorted_edges.end());

        for (size_t edge_id : sorted_edges)
        {
            const auto &edge = *mesh.getEdge(edge_id);
            size_t source = edge.getSource();
            size_t dest = edge.getDest();

            auto addFace = [&](size_t opposite_corner) {
                std::array<size_t, 3> face;
                face[0] = source;
                face[1] = dest;
                face[2] = opposite_corner;

                auto sorted_face = face;
                std::sort(sorted_face.begin(), sorted_face.end());
                if (nodes_anticlockwise(sorted_face))
                    std::swap(sorted_face[0], sorted_face[1]);
                faces.insert(sorted_face);
            };

            addFace(edge.payload.triangleOppositeNodes[0]);
            if (!edge.payload.border)
            {
                addFace(edge.payload.triangleOppositeNodes[1]);
            }
        }

        // Write faces
        std::vector<std::array<size_t, 3>> sorted_faces(faces.begin(), faces.end());
        std::sort(sorted_faces.begin(), sorted_faces.end());
        for (const auto &face : sorted_faces)
        {
            size_t v0 = node_to_index[face[0]];
            size_t v1 = node_to_index[face[1]];
            size_t v2 = node_to_index[face[2]];
            obj << "f " << v0 << "/" << v0 << " " << v1 << "/" << v1 << " " << v2 << "/" << v2 << "\n";
        }

        global_vertex_offset += sorted_nodes.size();
    }

    spdlog::info("Wrote textured OBJ: {}", obj_path);
}

} // namespace opencalibration::orthomosaic
