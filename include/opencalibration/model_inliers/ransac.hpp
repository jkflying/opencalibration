#pragma once

#include <opencalibration/model_inliers/essential_matrix_model.hpp>
#include <opencalibration/model_inliers/fundamental_matrix_model.hpp>
#include <opencalibration/model_inliers/homography_model.hpp>
#include <opencalibration/types/correspondence.hpp>
#include <opencalibration/types/feature_2d.hpp>
#include <opencalibration/types/feature_match.hpp>

#include <vector>

namespace opencalibration
{

template <typename Model>
double ransac(const std::vector<correspondence> &matches, Model &model, std::vector<bool> &inliers);

void assembleInliers(const std::vector<feature_match> &matches, const std::vector<bool> &inliers,
                     const std::vector<feature_2d> &source_features, const std::vector<feature_2d> &dest_features,
                     std::vector<feature_match_denormalized> &inlier_list);

} // namespace opencalibration
