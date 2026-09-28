#include <opencalibration/extract/extract_features.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <jk/KDTree.h>

namespace
{
using namespace opencalibration;

std::vector<feature_2d> to_features(const std::vector<cv::KeyPoint> &keypoints, const cv::Mat &descriptors,
                                    double scale)
{
    std::vector<feature_2d> features;
    features.reserve(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        feature_2d point;
        point.location = unscale_pixel({keypoints[i].pt.x, keypoints[i].pt.y}, scale);
        point.strength = keypoints[i].response;
        point.descriptor.reset();
        const uchar *row = &descriptors.at<uchar>(i, 0);
        for (int j = 0; j < feature_2d::DESCRIPTOR_BITS; j++)
            point.descriptor[j] = (row[j >> 3] >> (j & 7)) & 1;
        features.push_back(point);
    }
    return features;
}

extracted_features sparse_first_by_non_maximal_suppression(std::vector<feature_2d> features, double radius)
{
    std::sort(features.begin(), features.end(),
              [](const feature_2d &a, const feature_2d &b) -> bool { return a.strength > b.strength; });

    std::vector<feature_2d> sparse;
    std::vector<feature_2d> dense;

    auto toArray = [](const Eigen::Vector2d &v) -> std::array<double, 2> { return {v.x(), v.y()}; };
    jk::tree::KDTree<size_t, 2, 8> tree;
    if (!features.empty())
    {
        tree.addPoint(toArray(features[0].location), 0);
        sparse.push_back(features[0]);
    }

    auto searcher = tree.searcher();
    for (size_t i = 1; i < features.size(); i++)
    {
        const feature_2d &f = features[i];
        const auto &nearest = searcher.search(toArray(f.location), std::numeric_limits<double>::infinity(), 1);
        const double squared_distance = nearest[0].distance;
        if (squared_distance > radius * radius)
        {
            tree.addPoint(toArray(f.location), 0);
            sparse.push_back(f);
        }
        else
        {
            dense.push_back(f);
        }
    }

    const size_t num_sparse = sparse.size();
    sparse.insert(sparse.end(), std::make_move_iterator(dense.begin()), std::make_move_iterator(dense.end()));
    return {std::move(sparse), num_sparse};
}
} // namespace

namespace opencalibration
{

extracted_features extract_features(const cv::Mat &image)
{
    const double nms_scaled_pixel_radius = 8;

    if (image.empty())
    {
        return {};
    }

    cv::Mat image_scaled;
    cv::cvtColor(image, image_scaled, cv::COLOR_BGR2GRAY);
    const double scale =
        std::min(1.f, float(FEATURE_MAX_LENGTH_PIXELS) / std::max(image.size().width, image.size().height));
    cv::resize(image_scaled, image_scaled, cv::Size(0, 0), scale, scale, cv::INTER_AREA);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    auto akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, feature_2d::DESCRIPTOR_BITS, 3, 0.00005f);
    akaze->detectAndCompute(image_scaled, cv::noArray(), keypoints, descriptors);

    return sparse_first_by_non_maximal_suppression(to_features(keypoints, descriptors, scale),
                                                   nms_scaled_pixel_radius / scale);
}

} // namespace opencalibration
