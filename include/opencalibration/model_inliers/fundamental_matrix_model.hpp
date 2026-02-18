#pragma once

#include <opencalibration/types/correspondence.hpp>
#include <opencalibration/types/decomposed_pose.hpp>

#include <eigen3/Eigen/Geometry>

#include <array>
#include <stddef.h>
#include <vector>

namespace opencalibration
{

struct fundamental_matrix_model
{
    fundamental_matrix_model();

    static constexpr size_t MINIMUM_POINTS = 8;

    void fit(const std::vector<correspondence> &corrs, const std::array<size_t, MINIMUM_POINTS> &initial_indices);
    void fitInliers(const std::vector<correspondence> &corrs, const std::vector<bool> &inliers);

    size_t evaluate(const std::vector<correspondence> &corrs, std::vector<bool> &inliers);

    double error(const correspondence &cor);

    double inlier_threshold = 0.001;
    Eigen::Matrix3d fundamental_matrix;
};

} // namespace opencalibration
