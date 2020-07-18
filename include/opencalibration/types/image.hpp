#pragma once

#include <opencalibration/types/feature_2d.hpp>

#include <eigen3/Eigen/Geometry>

#include <memory>
#include <vector>

namespace opencalibration
{

struct image_metadata
{
};

struct image_pixeldata
{
};

struct image
{
    std::string path;

    image_metadata metadata;
    std::vector<feature_2d> descriptors;

    Eigen::Vector3d position {NAN, NAN, NAN};
    Eigen::Quaterniond orientation {NAN, NAN, NAN, NAN};
};
} // namespace opencalibration
