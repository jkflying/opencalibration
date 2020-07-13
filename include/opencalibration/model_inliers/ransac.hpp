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

    void fit(const std::array<correspondence, MINIMUM_POINTS> &corrs);

    double error(const correspondence &cor);

    Eigen::Matrix3d homography;
};

struct essential_matrix_model
{
    static constexpr size_t MINIMUM_POINTS = 5;

    void fit(const std::array<correspondence, MINIMUM_POINTS> &corrs);

    double error(const correspondence &cor);

    Eigen::Matrix3d essential_matrix;
};

struct fundamental_matrix_model
{
    static constexpr size_t MINIMUM_POINTS = 8;

    void fit(const std::array<correspondence, MINIMUM_POINTS> &corrs);

    double error(const correspondence &cor);

    Eigen::Matrix3d fundamental_matrix;
};

template <typename Model>
double ransac(const std::vector<correspondence> &matches, Model &model, std::vector<bool> &inliers);

} // namespace opencalibration
