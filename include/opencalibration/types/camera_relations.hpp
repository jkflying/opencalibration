#pragma once

#include <opencalibration/types/feature_match.hpp>

#include <eigen3/Eigen/Geometry>

#include <vector>

namespace opencalibration
{
    struct feature_match_denormalized
    {
        Eigen::Vector2d pixel_1, pixel_2;
    };

    struct camera_relations
    {
        std::vector<feature_match_denormalized> inlier_matches;

        Eigen::Matrix3d ransac_relation;
        enum class RelationType
        {
            HOMOGRAPHY
        } relationType = RelationType::HOMOGRAPHY;

        Eigen::Quaterniond relative_rotation{NAN,NAN,NAN,NAN};
    };
}
