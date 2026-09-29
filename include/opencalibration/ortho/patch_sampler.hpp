#pragma once

#include <opencalibration/types/camera_model.hpp>

#include <eigen3/Eigen/Core>
#include <opencv2/core.hpp>

#include <vector>

namespace opencalibration::orthomosaic
{

class PatchSampler
{
  public:
    static constexpr int MAX_PATCH_RADIUS = 16;

    static Eigen::Matrix2d computeJacobian(const Eigen::Vector3d &world_point,
                                           const DifferentiableCameraModel<double> &model,
                                           const Eigen::Vector3d &camera_position,
                                           const Eigen::Matrix3d &camera_orientation_inverse);

    struct BlockSample
    {
        Eigen::Vector2d pixel;
        cv::Vec3b *out;
    };

    void sampleBlock(const cv::Mat &bgr_image, const Eigen::Vector3d &reference_point,
                     const DifferentiableCameraModel<double> &model, const Eigen::Vector3d &camera_position,
                     const Eigen::Matrix3d &camera_orientation_inverse, double output_gsd,
                     const std::vector<BlockSample> &samples);

  private:
    cv::Mat _lab_roi;
    cv::Mat _lab_avg;
    cv::Mat _bgr_avg;
    std::vector<bool> _averaged;
};

} // namespace opencalibration::orthomosaic
