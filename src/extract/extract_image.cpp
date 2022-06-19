#include <opencalibration/extract/extract_image.hpp>

#include <opencalibration/extract/extract_features.hpp>
#include <opencalibration/extract/extract_metadata.hpp>
#include <opencalibration/performance/performance.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace
{
cv::Mat load_image(const std::string &path)
{
    if (cv::getNumThreads() != 1)
    {
        cv::setNumThreads(1);
    }

    return cv::imread(path);//, cv::IMREAD_IGNORE_ORIENTATION | cv::IMREAD_COLOR);
}
} // namespace

namespace opencalibration
{

image extract_image(const std::string &path)
{

    image img;
    img.path = path;

    PerformanceMeasure p("Load image");
    {
        const cv::Mat image = load_image(img.path);

        if (image.empty())
        {
            return img;
        }

        const double scale = 50 / std::sqrt(image.size().area());
        cv::resize(image, img.thumbnail, cv::Size(0, 0), scale, scale, cv::INTER_AREA);

        p.reset("Load features");
        img.features = extract_features(image);
    }

    p.reset("Load metadata");
    img.metadata = extract_metadata(img.path);

    img.model = std::make_shared<CameraModel>();

    img.model->focal_length_pixels = img.metadata.camera_info.focal_length_px;
    img.model->pixels_cols = img.metadata.camera_info.width_px;
    img.model->pixels_rows = img.metadata.camera_info.height_px;
    img.model->principle_point = Eigen::Vector2d(img.model->pixels_cols, img.model->pixels_rows) / 2;
    img.model->id = 0;

    return img;
}
} // namespace opencalibration
