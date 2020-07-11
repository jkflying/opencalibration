#include <opencalibration/oc_extract/extract_features.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
// #include <opencv2/opencv.hpp>

namespace opencalibration
{

std::vector<feature_2d> extract_features(const std::string &path)
{

    feature_2d sample;
    auto akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, feature_2d::DESCRIPTOR_BITS);

    cv::Mat image = cv::imread(path);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    akaze->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    std::vector<feature_2d> results;
    results.reserve(keypoints.size());

    for (size_t i = 0; i < keypoints.size(); i++)
    {
        feature_2d point;
        point.location.x() = keypoints[i].pt.x;
        point.location.y() = keypoints[i].pt.y;
        point.strength = keypoints[i].response;
        point.descriptor = *reinterpret_cast<std::bitset<feature_2d::DESCRIPTOR_BITS> *>(&descriptors.at<uchar>(i, 0));
    }

    return results;
}

} // namespace opencalibration
