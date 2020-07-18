#pragma once

#include <opencalibration/types/feature_2d.hpp>
#include <opencalibration/types/image_metadata.hpp>
#include <opencalibration/types/camera_model.hpp>

#include <eigen3/Eigen/Geometry>

#include <memory>
#include <vector>

namespace opencalibration
{

struct image_pixeldata
{
    std::unique_ptr<unsigned char> data;
};

struct image
{
    std::string path;

    // Loaded / processed from image data
    image_metadata metadata;
    std::vector<feature_2d> descriptors;

    // Things to discover and optimize
    CameraModel<> model;
    Eigen::Vector3d position {NAN, NAN, NAN};
    Eigen::Quaterniond orientation {NAN, NAN, NAN, NAN};
};
} // namespace opencalibration
