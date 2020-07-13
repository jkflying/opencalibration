#include <eigen3/Eigen/Geometry>
#include <opencalibration/distort/distort_keypoints.hpp>

namespace opencalibration
{
std::vector<correspondence> distort_keypoints(const std::vector<feature_2d> &features1,
                                              const std::vector<feature_2d> &features2,
                                              const std::vector<feature_match> &matches,
                                              const CameraModel<double> &model1, const CameraModel<double> &model2)
{
    std::vector<correspondence> distorted;
    distorted.reserve(matches.size());
    for (const feature_match &m : matches)
    {
        correspondence cor;
        cor.measurement1 = image_to_3d(features1[m.feature_index_1].location, model1);
        cor.measurement2 = image_to_3d(features2[m.feature_index_2].location, model2);
        distorted.push_back(cor);
    }

    return distorted;
}

Eigen::Vector3d image_to_3d(const Eigen::Vector2d &keypoint, const CameraModel<double> &model)
{
    // TODO: once we have real distortion, minimize:
    // Eigen::Vector2d error = keypoint - image_from_3d(guess, model);

    Eigen::Vector3d ray;
    switch (model.projection_type)
    {
    case CameraModel<double>::ProjectionType::PLANAR: {
        ray = ((keypoint - model.principle_point) / model.focal_length_pixels).homogeneous().normalized();
    }
    }
    return ray;
}
} // namespace opencalibration
