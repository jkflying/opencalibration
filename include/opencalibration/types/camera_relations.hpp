#pragma once

#include <opencalibration/types/decomposed_pose.hpp>
#include <opencalibration/types/feature_match.hpp>

#include <eigen3/Eigen/Geometry>

#include <vector>

namespace opencalibration
{

struct camera_relations
{
    std::vector<feature_match_denormalized> inlier_matches;
    std::vector<feature_match> matches;

    Eigen::Matrix3d ransac_relation = Eigen::Matrix3d::Constant(NAN);
    enum class RelationType
    {
        HOMOGRAPHY,
        UNKNOWN
    } relationType = RelationType::HOMOGRAPHY;

    std::array<decomposed_pose, 4> relative_poses;

    bool operator==(const camera_relations &other) const
    {
        return inlier_matches == other.inlier_matches && matches == other.matches &&
               ((ransac_relation.array().isNaN().all() && other.ransac_relation.array().isNaN().all()) ||
                ransac_relation == other.ransac_relation) &&
               relationType == other.relationType && relative_poses == other.relative_poses;
    }
};
} // namespace opencalibration
