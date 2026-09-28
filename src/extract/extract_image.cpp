#include <opencalibration/extract/extract_image.hpp>

#include <opencalibration/extract/camera_database.hpp>
#include <opencalibration/extract/extract_features.hpp>
#include <opencalibration/extract/extract_metadata.hpp>
#include <opencalibration/performance/performance.hpp>

#include <spdlog/spdlog.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <opencalibration/io/cv_raster_conversion.hpp>

namespace
{
int largest_decode_reduction_keeping_feature_resolution(const opencalibration::image_metadata &metadata)
{
    const size_t max_dim = std::max(metadata.camera_info.width_px, metadata.camera_info.height_px);
    size_t reduction = 1;
    while (reduction < 8 && max_dim / (reduction * 2) >= size_t(opencalibration::FEATURE_MAX_LENGTH_PIXELS))
    {
        reduction *= 2;
    }
    return static_cast<int>(reduction);
}

cv::Mat load_image(const std::string &path, int reduction)
{
    switch (reduction)
    {
    case 2:
        return cv::imread(path, cv::IMREAD_REDUCED_COLOR_2);
    case 4:
        return cv::imread(path, cv::IMREAD_REDUCED_COLOR_4);
    case 8:
        return cv::imread(path, cv::IMREAD_REDUCED_COLOR_8);
    default:
        return cv::imread(path);
    }
}
} // namespace

namespace opencalibration
{

std::optional<image> extract_image(const std::string &path)
{

    image img;
    img.path = path;

    PerformanceMeasure p("Load metadata");
    img.metadata = extract_metadata(img.path);

    p.reset("Load image");
    {
        const int reduction = largest_decode_reduction_keeping_feature_resolution(img.metadata);
        const cv::Mat image = load_image(img.path, reduction);

        if (image.empty())
        {
            return std::nullopt;
        }

        cv::Mat lab;
        cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);

        const double scale = 50 / std::sqrt(image.size().area());
        const cv::Size thumbnail_size(std::max(1, static_cast<int>(std::lround(image.cols * scale))),
                                      std::max(1, static_cast<int>(std::lround(image.rows * scale))));
        cv::Mat thumbnail_lab;
        cv::resize(lab, thumbnail_lab, thumbnail_size, 0, 0, cv::INTER_AREA);

        cv::Mat thumbnail;
        cv::cvtColor(thumbnail_lab, thumbnail, cv::COLOR_Lab2BGR);

        img.thumbnail = RasterToRGB(cvToRaster(thumbnail));

        p.reset("Load features");
        auto extracted = extract_features(image);
        img.features = std::move(extracted.features);
        img.num_sparse_features = extracted.num_sparse_features;
        for (feature_2d &f : img.features)
        {
            f.location = unscale_pixel(f.location, 1.0 / reduction);
        }
    }

    img.model = std::make_shared<CameraModel>();

    img.model->focal_length_pixels = img.metadata.camera_info.focal_length_px;
    img.model->pixels_cols = img.metadata.camera_info.width_px;
    img.model->pixels_rows = img.metadata.camera_info.height_px;
    img.model->principle_point = Eigen::Vector2d(img.model->pixels_cols, img.model->pixels_rows) / 2;

    static bool db_loaded = CameraDatabase::instance().load(CAMERA_DATABASE_PATH);
    (void)db_loaded;

    auto db_entry = CameraDatabase::instance().lookup(img.metadata.camera_info);
    if (db_entry.has_value())
    {
        applyDatabaseEntry(*db_entry, img.metadata.camera_info, *img.model);
        spdlog::debug("Applied camera database calibration for {} {}", img.metadata.camera_info.make,
                      img.metadata.camera_info.model);
    }

    img.model->id = 0;

    return img;
}
} // namespace opencalibration
