#pragma once

#include <opencalibration/types/feature_match.hpp>

#include <Eigen/Geometry>

#include <vector>

namespace opencalibration
{
struct feature_match_denormalized
{
    Eigen::Vector2d pixel_1, pixel_2;

    bool operator==(const feature_match_denormalized& other) const
    {
        return pixel_1 == other.pixel_1 && pixel_1 == other.pixel_2;
    }
};

struct camera_relations
{
    std::vector<feature_match_denormalized> inlier_matches;

    Eigen::Matrix3d ransac_relation;
    enum class RelationType
    {
        HOMOGRAPHY
    } relationType = RelationType::HOMOGRAPHY;

    Eigen::Quaterniond relative_rotation{NAN, NAN, NAN, NAN};
    Eigen::Vector3d relative_translation{NAN, NAN, NAN};

    bool operator==(const camera_relations& other) const
    {
        return inlier_matches == other.inlier_matches &&
               ransac_relation == other.ransac_relation &&
               relationType == other.relationType &&
               relative_rotation.coeffs() == other.relative_rotation.coeffs() &&
               relative_translation == other.relative_translation;
    }
};
} // namespace opencalibration
