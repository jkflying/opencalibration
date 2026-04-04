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
cv::Mat load_image(const std::string &path)
{
    return cv::imread(path); //, cv::IMREAD_IGNORE_ORIENTATION | cv::IMREAD_COLOR);
}
} // namespace

namespace opencalibration
{

std::optional<image> extract_image(const std::string &path)
{

    image img;
    img.path = path;

    PerformanceMeasure p("Load image");
    {
        const cv::Mat image = load_image(img.path);

        if (image.empty())
        {
            return std::nullopt;
        }

        cv::Mat lab;
        cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);

        const double scale = 50 / std::sqrt(image.size().area());
        cv::Mat thumbnail_lab;
        cv::resize(lab, thumbnail_lab, cv::Size(0, 0), scale, scale, cv::INTER_AREA);

        cv::Mat thumbnail;
        cv::cvtColor(thumbnail_lab, thumbnail, cv::COLOR_Lab2BGR);

        img.thumbnail = RasterToRGB(cvToRaster(thumbnail));

        p.reset("Load features");
        auto extracted = extract_features(image);
        img.features = std::move(extracted.features);
        img.num_sparse_features = extracted.num_sparse_features;
    }

    p.reset("Load metadata");
    img.metadata = extract_metadata(img.path);

    img.model = std::make_shared<CameraModel>();

    img.model->focal_length_pixels = img.metadata.camera_info.focal_length_px;
    img.model->pixels_cols = img.metadata.camera_info.width_px;
    img.model->pixels_rows = img.metadata.camera_info.height_px;
    img.model->principle_point = Eigen::Vector2d(img.model->pixels_cols, img.model->pixels_rows) / 2;

    // Load camera database on first call and apply calibration if available
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
