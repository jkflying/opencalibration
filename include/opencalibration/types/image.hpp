#pragma once

#include <opencalibration/types/camera_model.hpp>
#include <opencalibration/types/feature_2d.hpp>
#include <opencalibration/types/image_metadata.hpp>

#include <Eigen/Geometry>

#include <iostream>

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
    std::vector<feature_2d> features;

    // Things to discover and optimize
    CameraModel model;
    Eigen::Vector3d position{NAN, NAN, NAN};
    Eigen::Quaterniond orientation{NAN, NAN, NAN, NAN};

    bool operator==(const image &other) const
    {
        bool pat = path == other.path;
        bool met = metadata == other.metadata;
        bool feat = features == other.features;
        bool mod = model == other.model;
        bool pos =
            position == other.position || (position.array().isNaN().all() && other.position.array().isNaN().all());
        bool ori = orientation.coeffs() == other.orientation.coeffs() ||
                   (orientation.coeffs().array().isNaN().all() && other.orientation.coeffs().array().isNaN().all());

        return pat && met && feat && mod && pos && ori;
    }
};
} // namespace opencalibration
