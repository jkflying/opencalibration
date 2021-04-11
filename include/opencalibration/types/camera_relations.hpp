#pragma once

#include <opencalibration/types/feature_match.hpp>

#include <eigen3/Eigen/Geometry>

#include <vector>

namespace opencalibration
{
struct feature_match_denormalized
{
    Eigen::Vector2d pixel_1, pixel_2;

    bool operator==(const feature_match_denormalized &other) const
    {
        return pixel_1 == other.pixel_1 && pixel_2 == other.pixel_2;
    }
};

struct camera_relations
{
    std::vector<feature_match_denormalized> inlier_matches;

    Eigen::Matrix3d ransac_relation = Eigen::Matrix3d::Constant(NAN);
    enum class RelationType
    {
        HOMOGRAPHY,
        UNKNOWN
    } relationType = RelationType::HOMOGRAPHY;

    Eigen::Quaterniond relative_rotation{NAN, NAN, NAN, NAN};
    Eigen::Quaterniond relative_rotation2{NAN, NAN, NAN, NAN};
    Eigen::Vector3d relative_translation{NAN, NAN, NAN};
    Eigen::Vector3d relative_translation2{NAN, NAN, NAN};

    bool operator==(const camera_relations &other) const
    {

        return inlier_matches == other.inlier_matches &&
               (ransac_relation == other.ransac_relation ||
                (ransac_relation.array().isNaN().all() && other.ransac_relation.array().isNaN().all())) &&
               relationType == other.relationType &&
               (relative_rotation.coeffs() == other.relative_rotation.coeffs() ||
                (relative_rotation.coeffs().array().isNaN().all() &&
                 other.relative_rotation.coeffs().array().isNaN().all())) &&
               (relative_rotation2.coeffs() == other.relative_rotation2.coeffs() ||
                (relative_rotation2.coeffs().array().isNaN().all() &&
                 other.relative_rotation2.coeffs().array().isNaN().all())) &&
               (relative_translation == other.relative_translation ||
                (relative_translation.array().isNaN().all() && other.relative_translation.array().isNaN().all())) &&
               (relative_translation2 == other.relative_translation2 ||
                (relative_translation2.array().isNaN().all() && other.relative_translation2.array().isNaN().all()));
    }
};
} // namespace opencalibration
