#pragma once

#include <opencalibration/types/correspondence.hpp>
#include <opencalibration/types/decomposed_pose.hpp>

#include <eigen3/Eigen/Geometry>

#include <array>
#include <stddef.h>
#include <vector>

namespace opencalibration
{

struct essential_matrix_model
{
    essential_matrix_model();

    static constexpr size_t MINIMUM_POINTS = 5;

    void fit(const std::vector<correspondence> &corrs, const std::array<size_t, MINIMUM_POINTS> &initial_indices);
    void fitInliers(const std::vector<correspondence> &corrs, const std::vector<bool> &inliers);

    size_t evaluate(const std::vector<correspondence> &corrs, std::vector<bool> &inliers);

    double error(const correspondence &cor);

    bool decompose(const std::vector<correspondence> &corrs, const std::vector<bool> &inliers,
                   std::array<decomposed_pose, 4> &poses);

    double inlier_threshold{0.001};
    Eigen::Matrix3d essential_matrix;
};

} // namespace opencalibration
