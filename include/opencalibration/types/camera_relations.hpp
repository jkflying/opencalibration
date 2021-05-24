#pragma once

#include <opencalibration/types/decomposed_pose.hpp>
#include <opencalibration/types/feature_match.hpp>

#include <eigen3/Eigen/Geometry>

#include <vector>

namespace opencalibration
{
struct feature_match_denormalized
{
    Eigen::Vector2d pixel_1{NAN, NAN}, pixel_2{NAN, NAN};
    size_t feature_index_1, feature_index_2;

    bool operator==(const feature_match_denormalized &other) const
    {
        return ((pixel_1.array().isNaN().all() && other.pixel_1.array().isNaN().all()) || pixel_1 == other.pixel_1) &&
               ((pixel_2.array().isNaN().all() && other.pixel_2.array().isNaN().all()) || pixel_2 == other.pixel_2) &&
               feature_index_1 == other.feature_index_1 && feature_index_2 == other.feature_index_2;
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

    std::array<decomposed_pose, 4> relative_poses;

    bool operator==(const camera_relations &other) const
    {
        return inlier_matches == other.inlier_matches &&
               ((ransac_relation.array().isNaN().all() && other.ransac_relation.array().isNaN().all()) ||
                ransac_relation == other.ransac_relation) &&
               relationType == other.relationType && relative_poses == other.relative_poses;
    }
};
} // namespace opencalibration
