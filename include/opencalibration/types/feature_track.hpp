#pragma once

#include <eigen3/Eigen/Core>

#include <functional>

namespace opencalibration
{
struct NodeIdFeatureIndex
{
    size_t node_id, feature_index;
    bool operator==(const NodeIdFeatureIndex &nifi) const
    {
        return nifi.node_id == node_id && nifi.feature_index == feature_index;
    }

    bool operator<(const NodeIdFeatureIndex &nifi) const
    {
        return std::make_pair(node_id, feature_index) < std::make_pair(nifi.node_id, nifi.feature_index);
    }

    // hash operator
    std::size_t operator()(opencalibration::NodeIdFeatureIndex const &nifi) const
    {
        return nifi.node_id ^ nifi.feature_index; // random number XOR incrementing number is safe and fast
    }
};

struct FeatureTrack
{
    Eigen::Vector3d point{NAN, NAN, NAN};
    double error{NAN};
    std::vector<NodeIdFeatureIndex> measurements;
};

} // namespace opencalibration
