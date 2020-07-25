#include <opencalibration/extract/extract_features.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <jk/KDTree.h>

#include <iostream>
#include <mutex>

namespace opencalibration
{

std::vector<feature_2d> extract_features(const std::string &path)
{
    if (cv::getNumThreads() != 1)
    {
        cv::setNumThreads(1);
    }
    int max_length_pixels = 800;
    double nms_pixel_radius = 10;
    std::vector<feature_2d> results;

    cv::Mat image = cv::imread(path);
    if (image.size().width == 0 && image.size().height == 0)
    {
        return results;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    double scale = std::min(1.f, float(max_length_pixels) / std::max(image.size().width, image.size().height));
    cv::Mat image_scaled;
    cv::resize(image, image_scaled, cv::Size(0, 0), scale, scale);

    cv::Mat descriptors;

    // TODO: tuning

    std::vector<cv::KeyPoint> keypoints;

    auto akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, feature_2d::DESCRIPTOR_BITS, 3, 0.0001f);
    akaze->detectAndCompute(image_scaled, cv::noArray(), keypoints, descriptors);

    std::vector<feature_2d> oc_keypoints;
    oc_keypoints.reserve(keypoints.size());

    for (size_t i = 0; i < keypoints.size(); i++)
    {
        feature_2d point;
        point.location.x() = keypoints[i].pt.x / scale;
        point.location.y() = keypoints[i].pt.y / scale;
        point.strength = keypoints[i].response;
        point.descriptor = *reinterpret_cast<std::bitset<feature_2d::DESCRIPTOR_BITS> *>(&descriptors.at<uchar>(i, 0));
        oc_keypoints.push_back(point);
    }

    // non-maximal suppression (nearest-neighbor based)
    std::sort(oc_keypoints.begin(), oc_keypoints.end(),
              [](const feature_2d &a, const feature_2d &b) -> bool { return a.strength > b.strength; });

    results.reserve(std::min(keypoints.size(), static_cast<size_t>(image.size().width / nms_pixel_radius *
                                                                   image.size().height / nms_pixel_radius)));

    auto toArray = [](const Eigen::Vector2d &v) -> std::array<double, 2> { return {v.x(), v.y()}; };
    jk::tree::KDTree<size_t, 2, 8> tree;
    if (oc_keypoints.size() > 0)
    {
        tree.addPoint(toArray(oc_keypoints[0].location), 0);
        results.push_back(oc_keypoints[0]);
    }

    auto sqr = [](double d) { return d * d; };
    auto searcher = tree.searcher();
    for (const feature_2d &f : oc_keypoints)
    {
        const auto &nn = searcher.search(toArray(f.location), std::numeric_limits<double>::infinity(), 1);
        if (nn[0].distance * sqr(scale) > sqr(nms_pixel_radius))
        {
            tree.addPoint(toArray(f.location), 0);
            results.push_back(f);
        }
    }

    return results;
}

} // namespace opencalibration
