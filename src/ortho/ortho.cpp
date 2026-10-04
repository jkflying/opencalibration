#include <opencalibration/ortho/ortho.hpp>
#include <opencalibration/ortho/patch_sampler.hpp>

#include "thumbnail_encode.hpp"

#include <opencalibration/ortho/blending.hpp>
#include <opencalibration/ortho/color_balance.hpp>
#include <opencalibration/ortho/radiometric_cost.hpp>
#include <opencalibration/tile_ordering/tile_ordering.hpp>

#include <ceres/jet.h>
#include <cpl_string.h>
#include <eigen3/Eigen/Eigenvalues>
#include <jk/KDTree.h>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/geometry/utils.hpp>
#include <opencalibration/io/serialize.hpp>
#include <opencalibration/ortho/gdal_dataset.hpp>
#include <opencalibration/ortho/image_cache.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/surface/intersect.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <thread>
#include <type_traits>

namespace
{

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

constexpr double MAX_TAN_OFF_NADIR = 1.0;

bool withinNadirCone(const Eigen::Vector3d &camera, const Eigen::Vector3d &point)
{
    const double height = camera.z() - point.z();
    return height > 0 && (camera.head<2>() - point.head<2>()).norm() <= MAX_TAN_OFF_NADIR * height;
}

double coneHeightOrNan(jk::tree::KDTree<size_t, 2>::Searcher &cameras, const opencalibration::MeasurementGraph &graph,
                       double x, double y, double z)
{
    if (std::isnan(z))
        return z;
    const auto &nearest = cameras.search({x, y}, std::numeric_limits<double>::max(), 1);
    if (nearest.empty() || !withinNadirCone(graph.getNode(nearest[0].payload)->payload.position, {x, y, z}))
        return NAN;
    return z;
}

} // namespace

namespace opencalibration::orthomosaic
{

SampleGeometry sampleGeometry(const image &payload, const Eigen::Vector3d &world_point, const Eigen::Vector2d &pixel)
{
    const int cols = payload.model->pixels_cols;
    const int rows = payload.model->pixels_rows;
    SampleGeometry g;
    g.normalized_radius = normalizedImageRadius(pixel.x(), pixel.y(), cols, rows);
    std::tie(g.normalized_x, g.normalized_y) = normalizedImagePosition(pixel.x(), pixel.y(), cols, rows);

    const Eigen::Vector3d view_dir = (world_point - payload.position).normalized();
    const Eigen::Vector3d camera_down = payload.orientation * Eigen::Vector3d::UnitZ();
    g.view_angle_rad = static_cast<float>(std::acos(std::clamp(camera_down.dot(view_dir), -1.0, 1.0)));
    g.view_dir_x = static_cast<float>(view_dir.x());
    g.view_dir_y = static_cast<float>(view_dir.y());
    return g;
}

} // namespace opencalibration::orthomosaic

namespace
{

bool applyColorBalance(const opencalibration::orthomosaic::ColorBalanceResult &color_balance, size_t camera_id,
                       uint32_t model_id, const opencalibration::orthomosaic::SampleGeometry &geometry, cv::Vec3f &lab)
{
    using namespace opencalibration::orthomosaic;
    auto img_it = color_balance.per_image_params.find(camera_id);
    if (img_it == color_balance.per_image_params.end())
        return false;
    const auto &img_params = img_it->second;

    static const VignettingParams no_vignetting;
    auto mdl_it = color_balance.per_model_params.find(model_id);
    const auto &vig = mdl_it != color_balance.per_model_params.end() ? mdl_it->second : no_vignetting;

    const RadiometricModel<double> model{&img_params.log_cbrt_exposure,
                                         img_params.ab_offset.data(),
                                         &img_params.brdf_coeff,
                                         img_params.slope.data(),
                                         vig.log_cbrt_falloff_coeffs.data(),
                                         color_balance.horizontal_view_dir_log_cbrt_gain.data()};
    const double in[3] = {lab[0], lab[1], lab[2]};
    double out[3];
    correctRadiometry(in, model, geometry, out);
    lab =
        cv::Vec3f(std::clamp(out[0], 0.0, 100.0), std::clamp(out[1], -127.0, 127.0), std::clamp(out[2], -127.0, 127.0));
    return true;
}

cv::Vec3f rgbToLab(uint8_t r, uint8_t g, uint8_t b)
{
    cv::Mat pixel(1, 1, CV_32FC3, cv::Scalar(r / 255.f, g / 255.f, b / 255.f));
    cv::cvtColor(pixel, pixel, cv::COLOR_RGB2Lab);
    return pixel.at<cv::Vec3f>(0, 0);
}

cv::Vec3b labToRgb(const cv::Vec3f &lab)
{
    cv::Mat pixel(1, 1, CV_32FC3, cv::Scalar(lab[0], lab[1], lab[2]));
    cv::cvtColor(pixel, pixel, cv::COLOR_Lab2RGB);
    cv::Mat rgb;
    pixel.convertTo(rgb, CV_8UC3, 255.0);
    return rgb.at<cv::Vec3b>(0, 0);
}

} // namespace

namespace opencalibration::orthomosaic
{

Eigen::Matrix2d PatchSampler::computeJacobian(const Eigen::Vector3d &world_point,
                                              const DifferentiableCameraModel<double> &model,
                                              const Eigen::Vector3d &camera_position,
                                              const Eigen::Matrix3d &camera_orientation_inverse)
{
    using JetT = ceres::Jet<double, 2>;

    Eigen::Matrix<JetT, 3, 1> world_point_jet;
    world_point_jet[0] = JetT(world_point.x(), 0);
    world_point_jet[1] = JetT(world_point.y(), 1);
    world_point_jet[2] = JetT(world_point.z());

    DifferentiableCameraModel<JetT> model_jet;
    model_jet.focal_length_pixels = JetT(model.focal_length_pixels);
    model_jet.principle_point = model.principle_point.cast<JetT>();
    model_jet.radial_distortion = model.radial_distortion.cast<JetT>();
    model_jet.tangential_distortion = model.tangential_distortion.cast<JetT>();
    model_jet.pixels_cols = model.pixels_cols;
    model_jet.pixels_rows = model.pixels_rows;
    model_jet.projection_type = model.projection_type;

    Eigen::Matrix<JetT, 3, 1> camera_position_jet = camera_position.cast<JetT>();
    Eigen::Matrix<JetT, 3, 3> camera_orientation_inverse_jet = camera_orientation_inverse.cast<JetT>();

    Eigen::Matrix<JetT, 2, 1> pixel_jet =
        image_from_3d(world_point_jet, model_jet, camera_position_jet, camera_orientation_inverse_jet);

    Eigen::Matrix2d J;
    J(0, 0) = pixel_jet[0].v[0];
    J(0, 1) = pixel_jet[0].v[1];
    J(1, 0) = pixel_jet[1].v[0];
    J(1, 1) = pixel_jet[1].v[1];

    return J;
}

void PatchSampler::sampleBlock(const cv::Mat &bgr_image, const Eigen::Vector3d &reference_point,
                               const DifferentiableCameraModel<double> &model, const Eigen::Vector3d &camera_position,
                               const Eigen::Matrix3d &camera_orientation_inverse, double output_gsd,
                               const std::vector<BlockSample> &samples)
{
    auto nearest = [&](const Eigen::Vector2d &pixel) {
        return bgr_image.at<cv::Vec3b>(static_cast<int>(pixel.y()), static_cast<int>(pixel.x()));
    };

    Eigen::Matrix2d J = computeJacobian(reference_point, model, camera_position, camera_orientation_inverse);
    Eigen::Matrix2d M = output_gsd * output_gsd * J * J.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(M);
    const double a = std::sqrt(std::max(solver.eigenvalues()(1), 1e-6));
    const double b = std::sqrt(std::max(solver.eigenvalues()(0), 1e-6));

    if ((a < 1.0 && b < 1.0) || M.determinant() < 1e-12)
    {
        for (const auto &sample : samples)
            *sample.out = nearest(sample.pixel);
        return;
    }

    const int radius = std::min(static_cast<int>(std::ceil(a)), MAX_PATCH_RADIUS);
    const Eigen::Matrix2d M_inv = M.inverse();

    int x_min = bgr_image.cols - 1, y_min = bgr_image.rows - 1, x_max = 0, y_max = 0;
    for (const auto &sample : samples)
    {
        x_min = std::min(x_min, static_cast<int>(sample.pixel.x()) - radius);
        y_min = std::min(y_min, static_cast<int>(sample.pixel.y()) - radius);
        x_max = std::max(x_max, static_cast<int>(sample.pixel.x()) + radius);
        y_max = std::max(y_max, static_cast<int>(sample.pixel.y()) + radius);
    }
    x_min = std::max(0, x_min);
    y_min = std::max(0, y_min);
    x_max = std::min(bgr_image.cols - 1, x_max);
    y_max = std::min(bgr_image.rows - 1, y_max);

    cv::cvtColor(bgr_image(cv::Rect(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)), _lab_roi, cv::COLOR_BGR2Lab);

    _lab_avg.create(1, static_cast<int>(samples.size()), CV_8UC3);
    _averaged.assign(samples.size(), false);
    for (size_t i = 0; i < samples.size(); i++)
    {
        const Eigen::Vector2d &pixel = samples[i].pixel;
        const int cx = static_cast<int>(pixel.x());
        const int cy = static_cast<int>(pixel.y());

        double sum_L = 0, sum_a = 0, sum_b = 0;
        int count = 0;
        for (int py = std::max(y_min, cy - radius); py <= std::min(y_max, cy + radius); py++)
        {
            for (int px = std::max(x_min, cx - radius); px <= std::min(x_max, cx + radius); px++)
            {
                Eigen::Vector2d diff(px - pixel.x(), py - pixel.y());
                if (diff.transpose() * M_inv * diff <= 1.0)
                {
                    const cv::Vec3b &lab = _lab_roi.at<cv::Vec3b>(py - y_min, px - x_min);
                    sum_L += lab[0];
                    sum_a += lab[1];
                    sum_b += lab[2];
                    count++;
                }
            }
        }

        if (count == 0)
        {
            *samples[i].out = nearest(pixel);
            continue;
        }
        _lab_avg.at<cv::Vec3b>(0, static_cast<int>(i)) =
            cv::Vec3b(static_cast<uint8_t>(sum_L / count), static_cast<uint8_t>(sum_a / count),
                      static_cast<uint8_t>(sum_b / count));
        _averaged[i] = true;
    }

    cv::cvtColor(_lab_avg, _bgr_avg, cv::COLOR_Lab2BGR);
    for (size_t i = 0; i < samples.size(); i++)
        if (_averaged[i])
            *samples[i].out = _bgr_avg.at<cv::Vec3b>(0, static_cast<int>(i));
}

Eigen::Vector2d pixelCentre(const OrthoMosaicBounds &bounds, double gsd, double col, double row)
{
    return {bounds.min_x + (col + 0.5) * gsd, bounds.max_y - (row + 0.5) * gsd};
}

void rasterSizeCovering(const OrthoMosaicBounds &bounds, double gsd, int &width, int &height)
{
    auto pixels = [gsd](double extent) {
        const double n = std::ceil(extent / gsd);
        if (!std::isfinite(n))
            return 100;
        return std::max(1, static_cast<int>(n));
    };
    width = pixels(bounds.max_x - bounds.min_x);
    height = pixels(bounds.max_y - bounds.min_y);
}

uint64_t pixelCount(int width, int height)
{
    return static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
}

void coarsenGsdToFit(double &gsd, int &width, int &height, const OrthoMosaicBounds &bounds, uint64_t max_pixels)
{
    const bool size_depends_on_gsd =
        gsd > 0 && std::isfinite(bounds.max_x - bounds.min_x) && std::isfinite(bounds.max_y - bounds.min_y);
    if (!size_depends_on_gsd || max_pixels == 0)
        return;

    constexpr double kRoundUpMargin = 1 + 1e-6;
    for (uint64_t pixels = pixelCount(width, height); pixels > max_pixels; pixels = pixelCount(width, height))
    {
        gsd *= std::sqrt(static_cast<double>(pixels) / static_cast<double>(max_pixels)) * kRoundUpMargin;
        rasterSizeCovering(bounds, gsd, width, height);
    }
}

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

    const uint64_t output_pixels = pixelCount(width, height);
    if (output_pixels > total_input_pixels && total_input_pixels > 0)
    {
        coarsenGsdToFit(gsd, width, height, context.bounds, total_input_pixels);

        std::string stage_str = (stage_name && *stage_name) ? std::string(stage_name) + ": " : std::string("");
        spdlog::info("{}Clamped output resolution: GSD adjusted to {} (output pixels {} > input pixels {})", stage_str,
                     gsd, output_pixels, total_input_pixels);
    }
}

void clampOutputMegapixels(double &gsd, int &width, int &height, const OrthoMosaicBounds &bounds,
                           double max_output_megapixels, const char *stage_name = "")
{
    if (!std::isfinite(max_output_megapixels) || max_output_megapixels <= 0.0)
    {
        return;
    }

    const uint64_t output_pixels = pixelCount(width, height);
    const uint64_t max_output_pixels = static_cast<uint64_t>(max_output_megapixels * 1000000.0);
    if (max_output_pixels == 0 || output_pixels <= max_output_pixels)
    {
        return;
    }

    coarsenGsdToFit(gsd, width, height, bounds, max_output_pixels);

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

double arcPerPixel(const CameraModel &model)
{
    const double h = 0.001;
    Eigen::Vector2d pixel = image_from_3d({0, 0, 1}, model);
    Eigen::Vector2d pixelShift = image_from_3d({h, 0, 1}, model);
    return h / (pixel - pixelShift).norm();
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
        double arc_pixel = arcPerPixel(*payload.model);

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

    return context;
}

RayTraceContext::RayTraceContext(const std::vector<surface_model> &surfaces)
{
    for (const auto &surface : surfaces)
    {
        _searchers.emplace_back();
        if (!_searchers.back().init(surface.mesh))
        {
            spdlog::error("Could not initialize searcher on mesh surface");
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

namespace
{
struct ThumbnailSample
{
    size_t camera_id;
    uint32_t model_id;
    SampleGeometry geometry;
    cv::Vec3f lab;
};

struct ThumbnailColorSampling
{
    int pixel_step;
    size_t pairs_per_sample;
};

ThumbnailColorSampling thumbnailColorSampling(const OrthoMosaicContext &context, const MeasurementGraph &graph,
                                              const OrthoMosaicConfig &config)
{
    double full_resolution_gsd =
        calculateGSD(graph, context.involved_nodes, context.bounds.mean_surface_z, ImageResolution::FullResolution);
    int full_width, full_height;
    rasterSizeCovering(context.bounds, full_resolution_gsd, full_width, full_height);
    clampOutputResolution(full_resolution_gsd, full_width, full_height, context, graph, "Color sampling");
    clampOutputMegapixels(full_resolution_gsd, full_width, full_height, context.bounds, config.max_output_megapixels,
                          "Color sampling");

    const double full_res_samples_per_thumbnail_pixel =
        std::pow(context.gsd / (full_resolution_gsd * std::max(1, config.color_sample_spacing_full_res_px)), 2);
    if (full_res_samples_per_thumbnail_pixel >= 1)
        return {1, static_cast<size_t>(std::lround(full_res_samples_per_thumbnail_pixel))};
    return {static_cast<int>(std::lround(1 / std::sqrt(full_res_samples_per_thumbnail_pixel))), 1};
}

void appendCameraPairs(const std::vector<ThumbnailSample> &samples, size_t max_pairs,
                       std::vector<ColorCorrespondence> &correspondences)
{
    size_t pairs = 0;
    for (size_t b = 1; b < samples.size() && pairs < max_pairs; b++)
    {
        for (size_t a = 0; a < b && pairs < max_pairs; a++, pairs++)
        {
            const auto &sa = samples[a];
            const auto &sb = samples[b];
            correspondences.push_back({{sa.lab[0], sa.lab[1], sa.lab[2]},
                                       {sb.lab[0], sb.lab[1], sb.lab[2]},
                                       sa.camera_id,
                                       sb.camera_id,
                                       sa.model_id,
                                       sb.model_id,
                                       sa.geometry,
                                       sb.geometry});
        }
    }
}

ankerl::unordered_dense::map<size_t, CameraPosition> cameraPositions(
    const MeasurementGraph &graph, const ankerl::unordered_dense::set<size_t> &node_ids)
{
    ankerl::unordered_dense::map<size_t, CameraPosition> positions;
    for (size_t node_id : node_ids)
    {
        const auto &position = graph.getNode(node_id)->payload.position;
        positions[node_id] = {position.x(), position.y()};
    }
    return positions;
}
void applyThumbnailColorBalance(const ColorBalanceResult &balance, const std::vector<ThumbnailSample> &pixel_sources,
                                const Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic> &cameraUUID,
                                MultiLayerRaster<uint8_t> &pixelValues)
{
    constexpr uint32_t noSource = std::numeric_limits<uint32_t>::max();
#pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < cameraUUID.rows(); row++)
    {
        Eigen::Vector<uint8_t, Eigen::Dynamic> color(4);
        for (int col = 0; col < cameraUUID.cols(); col++)
        {
            if (static_cast<uint32_t>(cameraUUID(row, col)) == noSource)
                continue;
            ThumbnailSample source = pixel_sources[static_cast<size_t>(row) * cameraUUID.cols() + col];
            if (!applyColorBalance(balance, source.camera_id, source.model_id, source.geometry, source.lab))
                continue;
            const cv::Vec3b rgb = labToRgb(source.lab);
            color << rgb[0], rgb[1], rgb[2], 255;
            pixelValues.set(row, col, color);
        }
    }
}
} // namespace

OrthoMosaic generateOrthomosaic(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                                const OrthoMosaicConfig &config)
{
    OrthoMosaicContext context = prepareOrthoMosaicContext(surfaces, graph, ImageResolution::Thumbnail);

    spdlog::info("x range [{}; {}]  y range [{}; {}]  mean surface {}", context.bounds.min_x, context.bounds.max_x,
                 context.bounds.min_y, context.bounds.max_y, context.bounds.mean_surface_z);

    int width, height;
    rasterSizeCovering(context.bounds, context.gsd, width, height);
    clampOutputResolution(context.gsd, width, height, context, graph, "Thumbnail");
    const cv::Size image_dimensions(width, height);

    spdlog::info("gsd {}  img dims {}x{}", context.gsd, image_dimensions.width, image_dimensions.height);

    OrthoMosaic result;
    result.gsd = context.gsd;
    result.bounds = context.bounds;
    MultiLayerRaster<uint8_t> pixelValues(image_dimensions.height, image_dimensions.width, 4);
    pixelValues.layers[0].band = Band::RED;
    pixelValues.layers[1].band = Band::GREEN;
    pixelValues.layers[2].band = Band::BLUE;
    pixelValues.layers[3].band = Band::ALPHA;
    result.cameraUUID.pixels.resize(image_dimensions.height, image_dimensions.width);
    result.overlap.pixels.resize(image_dimensions.height, image_dimensions.width);
    result.dsm.pixels.resize(image_dimensions.height, image_dimensions.width);

    PerformanceMeasure p("Generate thumbnail");

    struct CameraCache
    {
        Eigen::Matrix3d inv_rotation;
        Eigen::Vector2d thumb_scale_xy;
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
        cc.thumb_scale_xy = {static_cast<double>(sz[1]) / payload.model->pixels_cols,
                             static_cast<double>(sz[0]) / payload.model->pixels_rows};
        camera_cache[node_id] = cc;
    }

    std::atomic<int> completed_rows{0};
    auto last_log_time = std::chrono::steady_clock::now();

    constexpr size_t maxOverlapCameras = 64;
    constexpr size_t colorCandidateCameras = 5;
    constexpr uint32_t noSource = std::numeric_limits<uint32_t>::max();

    const ThumbnailColorSampling color_sampling = thumbnailColorSampling(context, graph, config);
    spdlog::info("Color sampling: thumbnail sample step {} px, up to {} camera pairs per sample",
                 color_sampling.pixel_step, color_sampling.pairs_per_sample);

    std::vector<ThumbnailSample> pixel_sources(static_cast<size_t>(image_dimensions.height) * image_dimensions.width);
    std::vector<ColorCorrespondence> correspondences;

    const std::vector<MeshLineOfSight> surface_sightlines = sightlinesOver(surfaces);
#pragma omp parallel
    {
        RayTraceContext rayTrace(surfaces);
        auto sightlines = surface_sightlines;
        auto cameraSearcher = context.imageGPSLocations.searcher();
        const std::vector<jk::tree::KDTree<size_t, 2>::DistancePayload> noCameras;
        std::vector<ColorCorrespondence> local_correspondences;
        std::vector<ThumbnailSample> samples;

#pragma omp for schedule(dynamic)
        for (int row = 0; row < image_dimensions.height; row++)
        {
            for (int col = 0; col < image_dimensions.width; col++)
            {
                const Eigen::Vector2d centre = pixelCentre(context.bounds, context.gsd, col, row);
                const double x = centre.x(), y = centre.y();

                const double z =
                    coneHeightOrNan(cameraSearcher, graph, x, y, rayTrace.traceHeight(x, y, context.mean_camera_z));

                Eigen::Vector3d sample_point(x, y, z);

                Eigen::Vector<uint8_t, Eigen::Dynamic> color;
                color.resize(4);
                color.fill(0);
                uint32_t pixelSource = noSource;

                const auto &nearestCameras =
                    std::isnan(z)
                        ? noCameras
                        : cameraSearcher.search({x, y}, std::numeric_limits<double>::max(), maxOverlapCameras);
                uint8_t overlapCount = 0;
                samples.clear();

                for (size_t i = 0; i < nearestCameras.size(); i++)
                {
                    const auto &candidate = nearestCameras[i];
                    const auto &payload = graph.getNode(candidate.payload)->payload;
                    const auto &cc = camera_cache.at(candidate.payload);

                    Eigen::Vector3d camera_ray = cc.inv_rotation * (sample_point - payload.position);
                    if (camera_ray.z() <= 0 || !withinNadirCone(payload.position, sample_point))
                        continue;

                    Eigen::Vector2d pixel = image_from_3d(camera_ray, *payload.model);
                    const Eigen::Vector2d thumb_pixel = pixel.cwiseProduct(cc.thumb_scale_xy);
                    if (!thumb_pixel.allFinite())
                        continue;

                    const int px = static_cast<int>(std::floor(thumb_pixel.x()));
                    const int py = static_cast<int>(std::floor(thumb_pixel.y()));

                    if (px >= 0 && px < cc.thumb_size[1] && py >= 0 && py < cc.thumb_size[0] &&
                        surfaceVisibleFrom(sightlines, sample_point, payload.position))
                    {
                        overlapCount++;
                        Eigen::Vector<uint8_t, Eigen::Dynamic> pixelValue(3);
                        if (i < colorCandidateCameras && payload.thumbnail.get(py, px, pixelValue))
                        {
                            if (pixelSource == noSource)
                            {
                                color << pixelValue, 255;
                                pixelSource = candidate.payload & 0xFFFFFFFF;
                            }
                            samples.push_back({candidate.payload, static_cast<uint32_t>(payload.model->id),
                                               sampleGeometry(payload, sample_point, pixel),
                                               rgbToLab(pixelValue[0], pixelValue[1], pixelValue[2])});
                        }
                    }
                }

                if (!samples.empty())
                    pixel_sources[static_cast<size_t>(row) * image_dimensions.width + col] = samples.front();

                if (row % color_sampling.pixel_step == 0 && col % color_sampling.pixel_step == 0)
                    appendCameraPairs(samples, color_sampling.pairs_per_sample, local_correspondences);

                if (pixelSource == noSource)
                {
                    uint8_t grey = (row + col) % 2 == 0 ? 64 : 128;
                    color << grey, grey, grey, 0;
                }

                pixelValues.set(row, col, color);
                result.cameraUUID.pixels(row, col) = pixelSource;
                result.overlap.pixels(row, col) = overlapCount;
                result.dsm.pixels(row, col) = static_cast<float>(z);
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

#pragma omp critical
        correspondences.insert(correspondences.end(), local_correspondences.begin(), local_correspondences.end());
    }

    spdlog::info("Color balance: {} correspondences from thumbnail", correspondences.size());
    if (!correspondences.empty())
        result.color_balance = solveColorBalance(correspondences, cameraPositions(graph, context.involved_nodes));

    applyThumbnailColorBalance(result.color_balance, pixel_sources, result.cameraUUID.pixels, pixelValues);

    result.pixelValues = std::move(pixelValues);
    return result;
}

GDALDatasetPtr createGeoTIFF(const std::string &path, int width, int height, int bands, GDALDataType type,
                             int block_size, const OrthoMosaicBounds &bounds, double gsd, const std::string &wkt)
{
    GDALDriverH driver = GDALGetDriverByName("GTiff");
    if (!driver)
        throw std::runtime_error("GTiff driver not available");

    const bool rgba = bands == 4;
    const std::string block = std::to_string(block_size);
    char **options = nullptr;
    options = CSLSetNameValue(options, "TILED", "YES");
    options = CSLSetNameValue(options, "BLOCKXSIZE", block.c_str());
    options = CSLSetNameValue(options, "BLOCKYSIZE", block.c_str());
    options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
    const char *horizontal_differencing = "2", *floating_point_predictor = "3";
    options = CSLSetNameValue(options, "PREDICTOR",
                              GDALDataTypeIsFloating(type) ? floating_point_predictor : horizontal_differencing);
    options = CSLSetNameValue(options, "NUM_THREADS", "ALL_CPUS");
    options = CSLSetNameValue(options, "BIGTIFF", "IF_SAFER");
    options = CSLSetNameValue(options, "SPARSE_OK", "YES");
    if (rgba)
    {
        options = CSLSetNameValue(options, "PHOTOMETRIC", "RGB");
        options = CSLSetNameValue(options, "ALPHA", "YES");
    }

    GDALDatasetH dataset = GDALCreate(driver, path.c_str(), width, height, bands, type, options);
    CSLDestroy(options);
    if (!dataset)
        throw std::runtime_error("Failed to create GeoTIFF: " + path);
    GDALDatasetPtr ptr(dataset);

    double geotransform[6] = {bounds.min_x, gsd, 0, bounds.max_y, 0, -gsd};
    GDALSetGeoTransform(dataset, geotransform);
    if (!wkt.empty())
        GDALSetProjection(dataset, wkt.c_str());
    if (!rgba)
        GDALSetRasterNoDataValue(GDALGetRasterBand(dataset, 1), std::numeric_limits<float>::quiet_NaN());

    return ptr;
}

int blockSizeDividingTile(int tile_size)
{
    constexpr int kPreferredBlockSize = 512;
    constexpr int kTiffBlockMultiple = 16;
    const int block_size = std::gcd(tile_size, kPreferredBlockSize);
    return block_size % kTiffBlockMultiple == 0 ? block_size : kPreferredBlockSize;
}

template <typename T>
void writeInterleavedWindow(GDALDatasetH dataset, int x_offset, int y_offset, int width, int height,
                            const std::vector<T> &buffer)
{
    const int bands = GDALGetRasterCount(dataset);
    static_assert(std::is_same_v<T, uint8_t> || std::is_same_v<T, float>);
    constexpr GDALDataType type = std::is_same_v<T, float> ? GDT_Float32 : GDT_Byte;
    const int pixel_space = bands * sizeof(T);
    CPLErr err =
        GDALDatasetRasterIO(dataset, GF_Write, x_offset, y_offset, width, height, const_cast<T *>(buffer.data()), width,
                            height, type, bands, nullptr, pixel_space, pixel_space * width, sizeof(T));
    if (err != CE_None)
        throw std::runtime_error(std::string("Failed to write tile to ") + GDALGetDescription(dataset));
}

std::vector<float> computeDSMTile(int tile_x, int tile_y, int tile_size, const OrthoMosaicBounds &bounds, double gsd,
                                  int output_width, int output_height, const std::vector<surface_model> &surfaces,
                                  double mean_camera_z, const MeasurementGraph &graph,
                                  const jk::tree::KDTree<size_t, 2> &imageGPSLocations)
{
    int x_offset = tile_x * tile_size;
    int y_offset = tile_y * tile_size;
    int tile_width = std::min(tile_size, output_width - x_offset);
    int tile_height = std::min(tile_size, output_height - y_offset);

    std::vector<float> tile_buffer(tile_width * tile_height, std::numeric_limits<float>::quiet_NaN());

#pragma omp parallel
    {
        PerformanceMeasure thread_perf("DSM tile rows");

        RayTraceContext rayTrace(surfaces);
        jk::tree::KDTree<size_t, 2>::Searcher cameras(imageGPSLocations);

#pragma omp for schedule(dynamic)
        for (int local_row = 0; local_row < tile_height; local_row++)
        {
            for (int local_col = 0; local_col < tile_width; local_col++)
            {
                int global_col = x_offset + local_col;
                int global_row = y_offset + local_row;

                const Eigen::Vector2d centre = pixelCentre(bounds, gsd, global_col, global_row);
                const double x = centre.x(), y = centre.y();

                const double z = coneHeightOrNan(cameras, graph, x, y, rayTrace.traceHeight(x, y, mean_camera_z));

                int idx = local_row * tile_width + local_col;
                tile_buffer[idx] = static_cast<float>(z);
            }
        }
    }

    return tile_buffer;
}

ankerl::unordered_dense::set<size_t> findTileCameras(int tile_x, int tile_y, int tile_size,
                                                     const OrthoMosaicBounds &bounds, double gsd, int output_width,
                                                     int output_height,
                                                     const jk::tree::KDTree<size_t, 2> &imageGPSLocations,
                                                     int num_neighbors)
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
            const int global_col = x_offset + (tile_width - 1) * sx / (N - 1);
            const int global_row = y_offset + (tile_height - 1) * sy / (N - 1);
            const Eigen::Vector2d centre = pixelCentre(bounds, gsd, global_col, global_row);

            for (const auto &closest : searcher.search({centre.x(), centre.y()}, INFINITY, num_neighbors))
                camera_ids.insert(closest.payload);
        }
    }

    return camera_ids;
}

namespace
{

void loadImages(const ankerl::unordered_dense::set<size_t> &camera_ids, const opencalibration::MeasurementGraph &graph,
                opencalibration::orthomosaic::FullResolutionImageCache &image_cache)
{
    PerformanceMeasure thread_perf("Ortho Stage 1 - read");

    std::vector<size_t> ids(camera_ids.begin(), camera_ids.end());
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < ids.size(); i++) // NOLINT(modernize-loop-convert)
    {
        const auto *node = graph.getNode(ids[i]);
        if (node)
            image_cache.getImage(ids[i], node->payload.path);
    }
}

class LookaheadPrefetcher
{
  public:
    LookaheadPrefetcher(const std::vector<std::pair<int, int>> &tile_order,
                        const opencalibration::TileCameraMap &tile_cameras, int num_tiles_x,
                        const opencalibration::MeasurementGraph &graph,
                        opencalibration::orthomosaic::FullResolutionImageCache &image_cache)
        : graph_(graph), image_cache_(image_cache)
    {
        tile_offsets_.reserve(tile_order.size());
        for (const auto &[tile_x, tile_y] : tile_order)
        {
            tile_offsets_.push_back(cameras_.size());
            const auto &cameras = tile_cameras.at(static_cast<size_t>(tile_y) * num_tiles_x + tile_x);
            cameras_.insert(cameras_.end(), cameras.begin(), cameras.end());
        }
        next_ = cameras_.size();

        const size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
        for (size_t i = 0; i < num_threads; i++)
            threads_.emplace_back([this] { run(); });
    }

    LookaheadPrefetcher(const LookaheadPrefetcher &) = delete;
    LookaheadPrefetcher &operator=(const LookaheadPrefetcher &) = delete;

    ~LookaheadPrefetcher()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &thread : threads_)
            thread.join();
    }

    void startFrom(size_t position)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            next_ = tile_offsets_[position];
            blocked_ = false;
            generation_++;
        }
        cv_.notify_all();
    }

  private:
    void run()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true)
        {
            cv_.wait(lock, [&] { return stop_ || (!blocked_ && next_ < cameras_.size()); });
            if (stop_)
                return;
            const size_t cam = cameras_[next_++];
            const size_t generation = generation_;
            lock.unlock();

            bool cached = true;
            if (const auto *node = graph_.getNode(cam))
            {
                PerformanceMeasure thread_perf("Ortho Stage 1 - prefetch");
                cached = image_cache_.tryPrefetch(cam, node->payload.path);
            }

            lock.lock();
            if (!cached && generation == generation_)
                blocked_ = true;
        }
    }

    const opencalibration::MeasurementGraph &graph_;
    opencalibration::orthomosaic::FullResolutionImageCache &image_cache_;
    std::vector<size_t> cameras_;
    std::vector<size_t> tile_offsets_;

    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    bool blocked_ = false;
    size_t next_;
    size_t generation_ = 0;
    std::vector<std::thread> threads_;
};

bool projectIntoImage(const image &payload, const Eigen::Matrix3d &inv_rotation, const Eigen::Vector3d &world_point,
                      Eigen::Vector2d &pixel)
{
    if ((inv_rotation * (world_point - payload.position)).z() <= 0 || !withinNadirCone(payload.position, world_point))
        return false;

    pixel = image_from_3d(world_point, *payload.model, payload.position, inv_rotation);
    return pixel.x() >= 0 && pixel.x() < payload.model->pixels_cols && pixel.y() >= 0 &&
           pixel.y() < payload.model->pixels_rows;
}

struct BlockCamera
{
    size_t id = 0;
    const image *payload = nullptr;
    const Eigen::Matrix3d *inv_rotation = nullptr;
    cv::Mat full_image;
    bool image_fetched = false;
    Eigen::Vector3d reference_point;
    std::vector<PatchSampler::BlockSample> samples;
};

struct BlendSample
{
    int tile_pixel = 0;
    const BlockCamera *camera = nullptr;
    float weight = 0;
    SampleGeometry geometry;
    cv::Vec3b color_bgr;
};

constexpr size_t kMaxBlendCameras = 3;
constexpr size_t kBlockCandidates = 8;
constexpr size_t kPixelCandidates = 5;

std::vector<uint8_t> backgroundTile(int x_offset, int y_offset, int tile_width, int tile_height)
{
    std::vector<uint8_t> rgba(static_cast<size_t>(tile_width) * tile_height * 4, 0);
    for (int row = 0; row < tile_height; row++)
    {
        for (int col = 0; col < tile_width; col++)
        {
            const uint8_t grey = ((y_offset + row) + (x_offset + col)) % 2 == 0 ? 64 : 128;
            uint8_t *pixel = &rgba[(static_cast<size_t>(row) * tile_width + col) * 4];
            pixel[0] = pixel[1] = pixel[2] = grey;
        }
    }
    return rgba;
}

void blendSamplesInto(const std::vector<BlendSample> &samples, const ColorBalanceResult &color_balance,
                      std::vector<uint8_t> &rgba_tile)
{
    cv::Mat lab(1, static_cast<int>(samples.size()), CV_32FC3);
    for (size_t i = 0; i < samples.size(); i++)
    {
        const auto &bgr = samples[i].color_bgr;
        lab.at<cv::Vec3f>(static_cast<int>(i)) = cv::Vec3f(bgr[0], bgr[1], bgr[2]) / 255.f;
    }
    cv::cvtColor(lab, lab, cv::COLOR_BGR2Lab);

    std::vector<int> tile_pixels;
    cv::Mat blended(1, static_cast<int>(samples.size()), CV_32FC3);
    for (size_t begin = 0; begin < samples.size();)
    {
        cv::Vec3f weighted_sum(0, 0, 0);
        float weight_sum = 0;
        size_t end = begin;
        for (; end < samples.size() && samples[end].tile_pixel == samples[begin].tile_pixel; end++)
        {
            const auto &sample = samples[end];
            cv::Vec3f &sample_lab = lab.at<cv::Vec3f>(static_cast<int>(end));
            applyColorBalance(color_balance, sample.camera->id, sample.camera->payload->model->id, sample.geometry,
                              sample_lab);
            weighted_sum += sample.weight * sample_lab;
            weight_sum += sample.weight;
        }
        blended.at<cv::Vec3f>(static_cast<int>(tile_pixels.size())) = weighted_sum / weight_sum;
        tile_pixels.push_back(samples[begin].tile_pixel);
        begin = end;
    }

    blended = blended.colRange(0, static_cast<int>(tile_pixels.size()));
    cv::cvtColor(blended, blended, cv::COLOR_Lab2RGB);
    cv::Mat rgb;
    blended.convertTo(rgb, CV_8UC3, 255.0);
    for (size_t i = 0; i < tile_pixels.size(); i++)
    {
        const auto &color = rgb.at<cv::Vec3b>(static_cast<int>(i));
        uint8_t *pixel = &rgba_tile[static_cast<size_t>(tile_pixels[i]) * 4];
        pixel[0] = color[0];
        pixel[1] = color[1];
        pixel[2] = color[2];
        pixel[3] = 255;
    }
}

TileUpdate tileUpdate(const std::vector<uint8_t> &rgba, int x_offset, int y_offset, int tile_width, int tile_height,
                      int output_width, int output_height, int tile_index, int total_tiles,
                      const OrthoMosaicBounds &bounds, double gsd)
{
    const int scale = std::max(1, (std::max(tile_width, tile_height) + 127) / 128);
    const int thumb_w = (tile_width + scale - 1) / scale;
    const int thumb_h = (tile_height + scale - 1) / scale;

    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> blue(thumb_h, thumb_w), green(thumb_h, thumb_w),
        red(thumb_h, thumb_w), alpha(thumb_h, thumb_w);
    for (int ty = 0; ty < thumb_h; ty++)
    {
        for (int tx = 0; tx < thumb_w; tx++)
        {
            const size_t src = static_cast<size_t>(std::min(ty * scale, tile_height - 1)) * tile_width +
                               std::min(tx * scale, tile_width - 1);
            const bool valid = rgba[src * 4 + 3] > 0;
            red(ty, tx) = valid ? rgba[src * 4 + 0] : 0;
            green(ty, tx) = valid ? rgba[src * 4 + 1] : 0;
            blue(ty, tx) = valid ? rgba[src * 4 + 2] : 0;
            alpha(ty, tx) = rgba[src * 4 + 3];
        }
    }

    TileUpdate tu;
    tu.pixel_x = x_offset;
    tu.pixel_y = y_offset;
    tu.pixel_w = tile_width;
    tu.pixel_h = tile_height;
    tu.total_output_width = output_width;
    tu.total_output_height = output_height;
    tu.tile_index = tile_index;
    tu.total_tiles = total_tiles;
    tu.thumbnail.png_base64 = encodeThumbnailToBase64PNG(blue, green, red, alpha);
    tu.thumbnail.bounds_min_x = bounds.min_x;
    tu.thumbnail.bounds_max_y = bounds.max_y;
    tu.thumbnail.meters_per_pixel = gsd;
    return tu;
}

void buildOverviews(GDALDatasetH dataset, int width, int height)
{
    constexpr int kGdaladdoMinOverviewSize = 256;
    std::vector<int> overview_levels;
    for (int level = 2; std::max(width, height) / level >= kGdaladdoMinOverviewSize; level *= 2)
        overview_levels.push_back(level);
    if (overview_levels.empty())
        return;

    PerformanceMeasure p("Ortho - overviews");
    const auto start = std::chrono::steady_clock::now();
    GDALFlushCache(dataset);
    CPLSetThreadLocalConfigOption("GDAL_NUM_THREADS", "ALL_CPUS");
    CPLErr err = GDALBuildOverviews(dataset, "AVERAGE", static_cast<int>(overview_levels.size()),
                                    overview_levels.data(), 0, nullptr, nullptr, nullptr);
    CPLSetThreadLocalConfigOption("GDAL_NUM_THREADS", nullptr);
    if (err != CE_None)
        spdlog::warn("Failed to build overviews for {}", GDALGetDescription(dataset));
    else
        spdlog::info("Built {} overview levels for {} in {:.1f}s", overview_levels.size(), GDALGetDescription(dataset),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

} // namespace

std::vector<uint8_t> renderTile(int tile_x, int tile_y, int tile_size, const OrthoMosaicBounds &bounds, double gsd,
                                int output_width, int output_height, const std::vector<float> &dsm_tile,
                                const std::vector<MeshLineOfSight> &sightlines, const MeasurementGraph &graph,
                                const jk::tree::KDTree<size_t, 2> &imageGPSLocations,
                                const ankerl::unordered_dense::map<size_t, Eigen::Matrix3d> &inv_rotation_cache,
                                FullResolutionImageCache &image_cache, const ColorBalanceResult &color_balance,
                                double feather_distance)
{
    int x_offset = tile_x * tile_size;
    int y_offset = tile_y * tile_size;
    int tile_width = std::min(tile_size, output_width - x_offset);
    int tile_height = std::min(tile_size, output_height - y_offset);

    std::vector<uint8_t> rgba_tile = backgroundTile(x_offset, y_offset, tile_width, tile_height);

    constexpr int kBlockSize = 8;
    const int blocks_x = (tile_width + kBlockSize - 1) / kBlockSize;
    const int blocks_y = (tile_height + kBlockSize - 1) / kBlockSize;

    auto localPixelCentre = [&](double local_col, double local_row) {
        return pixelCentre(bounds, gsd, x_offset + local_col, y_offset + local_row);
    };

#pragma omp parallel
    {
        PerformanceMeasure thread_perf("Ortho - render");

        PatchSampler sampler;
        auto local_sightlines = sightlines;

        jk::tree::KDTree<size_t, 2>::Searcher tree_searcher(imageGPSLocations);
        ankerl::unordered_dense::map<uint64_t, cv::Mat> local_image_cache;
        auto fetchImage = [&](const BlockCamera &cam) {
            auto it = local_image_cache.find(cam.id);
            if (it != local_image_cache.end())
                return it->second;
            cv::Mat full_image = image_cache.getImage(cam.id, cam.payload->path);
            if (!full_image.empty())
                local_image_cache.emplace(cam.id, full_image);
            return full_image;
        };

        std::vector<BlockCamera> cameras;
        std::vector<size_t> rank;
        std::vector<BlendSample> samples;
        samples.reserve(kBlockSize * kBlockSize * kMaxBlendCameras);

#pragma omp for schedule(dynamic)
        for (int block = 0; block < blocks_x * blocks_y; block++)
        {
            const int row_begin = (block / blocks_x) * kBlockSize;
            const int col_begin = (block % blocks_x) * kBlockSize;
            const int row_end = std::min(row_begin + kBlockSize, tile_height);
            const int col_end = std::min(col_begin + kBlockSize, tile_width);

            cameras.clear();
            samples.clear();
            const Eigen::Vector2d block_centre =
                localPixelCentre(0.5 * (col_begin + col_end - 1), 0.5 * (row_begin + row_end - 1));
            const auto &candidates =
                tree_searcher.search({block_centre.x(), block_centre.y()}, INFINITY, kBlockCandidates);
            for (const auto &candidate : candidates)
            {
                auto &cam = cameras.emplace_back();
                cam.id = candidate.payload;
                cam.payload = &graph.getNode(candidate.payload)->payload;
                auto inv_rot_it = inv_rotation_cache.find(candidate.payload);
                if (inv_rot_it != inv_rotation_cache.end())
                    cam.inv_rotation = &inv_rot_it->second;
            }

            for (int local_row = row_begin; local_row < row_end; local_row++)
            {
                for (int local_col = col_begin; local_col < col_end; local_col++)
                {
                    const float z = dsm_tile[local_row * tile_width + local_col];
                    if (std::isnan(z))
                        continue;
                    Eigen::Vector3d sample_point;
                    sample_point << localPixelCentre(local_col, local_row), z;

                    rank.resize(cameras.size());
                    std::iota(rank.begin(), rank.end(), 0);
                    auto distance2 = [&](size_t i) {
                        return (cameras[i].payload->position.head<2>() - sample_point.head<2>()).squaredNorm();
                    };
                    std::sort(rank.begin(), rank.end(),
                              [&](size_t l, size_t r) { return distance2(l) < distance2(r); });

                    size_t num_blended = 0;
                    double nearest_distance = 0;
                    for (size_t r = 0; r < std::min(rank.size(), kPixelCandidates) && num_blended < kMaxBlendCameras;
                         r++)
                    {
                        auto &cam = cameras[rank[r]];
                        if (cam.inv_rotation == nullptr)
                            continue;

                        Eigen::Vector2d pixel;
                        if (!projectIntoImage(*cam.payload, *cam.inv_rotation, sample_point, pixel) ||
                            !surfaceVisibleFrom(local_sightlines, sample_point, cam.payload->position))
                            continue;

                        const double distance = std::sqrt(distance2(rank[r]));
                        if (num_blended == 0)
                            nearest_distance = distance;
                        const double feather =
                            num_blended == 0 ? 1.0 : 1.0 - (distance - nearest_distance) / feather_distance;
                        if (!(feather > 0))
                            break;

                        if (!cam.image_fetched)
                        {
                            cam.full_image = fetchImage(cam);
                            cam.image_fetched = true;
                        }
                        if (cam.full_image.empty() || pixel.x() >= cam.full_image.cols ||
                            pixel.y() >= cam.full_image.rows)
                            continue;

                        auto &sample = samples.emplace_back();
                        sample.tile_pixel = local_row * tile_width + local_col;
                        sample.camera = &cam;
                        sample.geometry = sampleGeometry(*cam.payload, sample_point, pixel);
                        sample.weight =
                            static_cast<float>(feather) *
                            computeBlendWeight(static_cast<float>(pixel.x()), static_cast<float>(pixel.y()),
                                               cam.payload->model->pixels_cols, cam.payload->model->pixels_rows,
                                               static_cast<float>((sample_point - cam.payload->position).norm()));

                        if (cam.samples.empty())
                            cam.reference_point = sample_point;
                        cam.samples.push_back({pixel, &sample.color_bgr});
                        num_blended++;
                    }
                }
            }

            for (const auto &cam : cameras)
            {
                if (!cam.samples.empty())
                    sampler.sampleBlock(cam.full_image, cam.reference_point, *cam.payload->model, cam.payload->position,
                                        *cam.inv_rotation, gsd, cam.samples);
            }

            if (!samples.empty())
                blendSamplesInto(samples, color_balance, rgba_tile);
        }
    }

    return rgba_tile;
}

void generateGeoTIFF(const std::vector<surface_model> &surfaces, const MeasurementGraph &graph,
                     const GeoCoord &coord_system, const ColorBalanceResult &color_balance,
                     const std::string &output_path, const std::string &dsm_output_path,
                     const OrthoMosaicConfig &config, TileProgressCallback tile_progress)
{
    spdlog::info("Generating orthomosaic GeoTIFF: {}", output_path);
    PerformanceMeasure p("Ortho - setup");

    GDALAllRegister();

    OrthoMosaicContext context = prepareOrthoMosaicContext(surfaces, graph, ImageResolution::FullResolution);

    int width, height;
    rasterSizeCovering(context.bounds, context.gsd, width, height);

    clampOutputResolution(context.gsd, width, height, context, graph, "Orthomosaic");
    clampOutputMegapixels(context.gsd, width, height, context.bounds, config.max_output_megapixels, "Orthomosaic");

    const OrthoMosaicBounds &bounds = context.bounds;
    double gsd = context.gsd;
    const double feather_distance = 2.0 * config.blend_transition_radius * gsd;

    spdlog::info("GSD: {}  Output dimensions: {}x{} pixels", gsd, width, height);

    const int tile_size = config.tile_size;
    const int block_size = blockSizeDividingTile(tile_size);
    const std::string wkt = coord_system.getWKT();
    GDALDatasetPtr output_ds;
    if (!output_path.empty())
        output_ds = createGeoTIFF(output_path, width, height, 4, GDT_Byte, block_size, bounds, gsd, wkt);
    GDALDatasetPtr dsm_ds;
    if (!dsm_output_path.empty())
        dsm_ds = createGeoTIFF(dsm_output_path, width, height, 1, GDT_Float32, block_size, bounds, gsd, wkt);

    ankerl::unordered_dense::map<size_t, Eigen::Matrix3d> inv_rotation_cache;
    for (size_t node_id : context.involved_nodes)
    {
        const auto *node = graph.getNode(node_id);
        if (node)
        {
            inv_rotation_cache[node_id] = node->payload.orientation.inverse().toRotationMatrix();
        }
    }
    const std::vector<MeshLineOfSight> sightlines = sightlinesOver(surfaces);

    int num_tiles_x = (width + tile_size - 1) / tile_size;
    int num_tiles_y = (height + tile_size - 1) / tile_size;
    int total_tiles = num_tiles_x * num_tiles_y;

    spdlog::info("Processing {} tiles ({}x{} grid)", total_tiles, num_tiles_x, num_tiles_y);

    int completed_tiles = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_log_time = start_time;

    p.reset("");

    TileCameraMap tile_camera_map;
    for (int ty = 0; ty < num_tiles_y; ty++)
        for (int tx = 0; tx < num_tiles_x; tx++)
        {
            size_t tile_idx = static_cast<size_t>(ty) * num_tiles_x + tx;
            tile_camera_map[tile_idx] = findTileCameras(tx, ty, tile_size, bounds, gsd, width, height,
                                                        context.imageGPSLocations, kPixelCandidates);
        }

    const size_t image_cache_size = computeImageCacheSize(tile_camera_map);
    FullResolutionImageCache image_cache(image_cache_size);
    const auto tile_order = hilbertTileOrder(num_tiles_x, num_tiles_y);

    const ImageUseSchedule image_schedule(tile_order, tile_camera_map, num_tiles_x);
    std::atomic<size_t> current_tile_position{0};
    image_cache.setNextUse([&](size_t node_id) { return image_schedule.nextUse(node_id, current_tile_position); });

    LookaheadPrefetcher prefetcher(tile_order, tile_camera_map, num_tiles_x, graph, image_cache);

    std::future<void> write_future;

    double dsm_s = 0, load_s = 0, process_s = 0, write_wait_s = 0;
    auto lap = [t = std::chrono::steady_clock::now()](double &total) mutable {
        auto now = std::chrono::steady_clock::now();
        total += std::chrono::duration<double>(now - t).count();
        t = now;
    };

    for (size_t i = 0; i < tile_order.size(); i++)
    {
        const auto tile_x = tile_order[i].first;
        const auto tile_y = tile_order[i].second;
        current_tile_position = i;
        prefetcher.startFrom(i);

        std::vector<float> dsm_tile = computeDSMTile(tile_x, tile_y, tile_size, bounds, gsd, width, height, surfaces,
                                                     context.mean_camera_z, graph, context.imageGPSLocations);
        lap(dsm_s);

        std::vector<uint8_t> rgba_tile;
        if (output_ds)
        {
            loadImages(tile_camera_map.at(static_cast<size_t>(tile_y) * num_tiles_x + tile_x), graph, image_cache);
            lap(load_s);
            rgba_tile =
                renderTile(tile_x, tile_y, tile_size, bounds, gsd, width, height, dsm_tile, sightlines, graph,
                           context.imageGPSLocations, inv_rotation_cache, image_cache, color_balance, feather_distance);
        }

        const int x_off = tile_x * tile_size;
        const int y_off = tile_y * tile_size;
        const int tw = std::min(tile_size, width - x_off);
        const int th = std::min(tile_size, height - y_off);

        if (tile_progress && output_ds)
            tile_progress(tileUpdate(rgba_tile, x_off, y_off, tw, th, width, height, completed_tiles + 1, total_tiles,
                                     bounds, gsd));
        lap(process_s);

        if (write_future.valid())
            write_future.get();
        lap(write_wait_s);

        auto rgba_tile_ptr = std::make_shared<std::vector<uint8_t>>(std::move(rgba_tile));
        auto dsm_tile_ptr = std::make_shared<std::vector<float>>(std::move(dsm_tile));
        write_future = std::async(std::launch::async, [&, rgba_tile_ptr, dsm_tile_ptr, x_off, y_off, tw, th] {
            PerformanceMeasure thread_perf("Ortho - write");
            if (output_ds)
                writeInterleavedWindow(output_ds.get(), x_off, y_off, tw, th, *rgba_tile_ptr);
            if (dsm_ds)
                writeInterleavedWindow(dsm_ds.get(), x_off, y_off, tw, th, *dsm_tile_ptr);
        });

        completed_tiles++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() >= 5 ||
            completed_tiles == total_tiles)
        {
            double progress = 100.0 * completed_tiles / total_tiles;
            spdlog::info("Orthomosaic progress: {:.1f}% ({}/{} tiles, {} seconds), time in dsm {:.0f}s, "
                         "image load {:.0f}s, process {:.0f}s, write wait {:.0f}s",
                         progress, completed_tiles, total_tiles, elapsed, dsm_s, load_s, process_s, write_wait_s);
            last_log_time = now;
        }
    }

    if (write_future.valid())
        write_future.get();

    spdlog::info("Building overviews...");
    if (output_ds)
        buildOverviews(output_ds.get(), width, height);
    if (dsm_ds)
        buildOverviews(dsm_ds.get(), width, height);

    spdlog::info("Orthomosaic complete: {}", output_path);
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

    std::string base_path = obj_path;
    if (base_path.size() >= 4 && base_path.substr(base_path.size() - 4) == ".obj")
    {
        base_path = base_path.substr(0, base_path.size() - 4);
    }
    std::string mtl_path = base_path + ".mtl";
    std::string jpg_path = base_path + ".jpg";

    auto filename_only = [](const std::string &path) {
        size_t pos = path.find_last_of("/\\");
        return (pos != std::string::npos) ? path.substr(pos + 1) : path;
    };
    std::string mtl_filename = filename_only(mtl_path);
    std::string jpg_filename = filename_only(jpg_path);

    if (ds.GetRasterCount() < 3)
    {
        spdlog::error("Expected RGB bands in {}", geotiff_path);
        return;
    }
    int rgb_bands_in_bgr_order[3] = {3, 2, 1};
    constexpr int MAX_TEXTURE_DIM = 16384;
    const double tex_scale = std::min(1.0, static_cast<double>(MAX_TEXTURE_DIM) / std::max(img_width, img_height));
    const int tex_width = std::max(1, static_cast<int>(std::lround(img_width * tex_scale)));
    const int tex_height = std::max(1, static_cast<int>(std::lround(img_height * tex_scale)));
    cv::Mat texture(tex_height, tex_width, CV_8UC3);
    GDALRasterIOExtraArg extra_arg;
    INIT_RASTERIO_EXTRA_ARG(extra_arg);
    extra_arg.eResampleAlg = GRIORA_Average;
    if (GDALDatasetRasterIOEx(dataset.get(), GF_Read, 0, 0, img_width, img_height, texture.data, tex_width, tex_height,
                              GDT_Byte, 3, rgb_bands_in_bgr_order, 3, static_cast<GSpacing>(texture.step), 1,
                              &extra_arg) != CE_None)
    {
        spdlog::error("Failed to read texture from {}", geotiff_path);
        return;
    }
    if (!cv::imwrite(jpg_path, texture))
    {
        spdlog::error("Failed to write texture: {}", jpg_path);
        return;
    }
    spdlog::info("Wrote texture: {} ({}x{})", jpg_path, texture.cols, texture.rows);

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

    for (const auto &surface : surfaces)
    {
        const auto &mesh = surface.mesh;
        if (mesh.size_edges() == 0)
            continue;

        std::vector<size_t> sorted_nodes;
        sorted_nodes.reserve(mesh.size_nodes());
        std::transform(mesh.cnodebegin(), mesh.cnodeend(), std::back_inserter(sorted_nodes),
                       [](const auto &iter) { return iter.first; });
        std::sort(sorted_nodes.begin(), sorted_nodes.end());

        std::unordered_map<size_t, size_t> node_to_index;

        for (size_t node_id : sorted_nodes)
        {
            const auto &loc = mesh.getNode(node_id)->payload.location;
            node_to_index[node_id] = global_vertex_offset + node_to_index.size() + 1; // 1-based OBJ indices

            obj << "v " << loc.x() << " " << loc.y() << " " << loc.z() << "\n";

            double u = (loc.x() - min_x) / extent_x;
            double v = 1.0 - (max_y - loc.y()) / extent_y;
            obj << "vt " << u << " " << v << "\n";
        }

        for (const auto &face : meshFaces(mesh))
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
