#pragma once

#include <opencalibration/types/correspondence.hpp>
#include <opencalibration/types/feature_match.hpp>

#include <eigen3/Eigen/Core>

#include <array>
#include <random>
#include <stddef.h>
#include <vector>

namespace opencalibration
{

struct homography_model
{
    static constexpr size_t MINIMUM_POINTS = 4;

    void fit(const std::vector<correspondence> &corrs, const std::array<size_t, MINIMUM_POINTS> &initial_indices);
    void fitInliers(const std::vector<correspondence> &corrs, const std::vector<bool> &inliers);

    size_t evaluate(const std::vector<correspondence> &corrs, std::vector<bool> &inliers);

    double error(const correspondence &cor);

    double inlier_threshold = 0.02;
    Eigen::Matrix3d homography;
};

template <typename Model>
double ransac(const std::vector<correspondence> &matches, Model &model, std::vector<bool> &inliers);

} // namespace opencalibration
