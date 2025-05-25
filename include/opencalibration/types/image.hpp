#pragma once

#include <opencalibration/types/camera_model.hpp>
#include <opencalibration/types/feature_2d.hpp>
#include <opencalibration/types/image_metadata.hpp>
#include <opencalibration/types/raster.hpp>

#include <eigen3/Eigen/Geometry>

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace opencalibration
{

struct image
{
    std::string path;

    // Loaded / processed from image data
    image_metadata metadata;
    std::vector<feature_2d> features;
    RGBRaster thumbnail;

    // Things to discover and optimize
    std::shared_ptr<CameraModel> model;
    Eigen::Vector3d position{NAN, NAN, NAN};
    Eigen::Quaterniond orientation{NAN, NAN, NAN, NAN};

    bool operator==(const image &other) const
    {
        bool pat = path == other.path;
        bool thu = thumbnail == other.thumbnail;
        bool met = metadata == other.metadata;
        bool feat = features == other.features;
        bool mod = (model == other.model) || (model != nullptr && other.model != nullptr && *model == *other.model);
        bool pos =
            (position.array().isNaN().all() && other.position.array().isNaN().all()) || position == other.position;
        bool ori = (orientation.coeffs().array().isNaN().all() && other.orientation.coeffs().array().isNaN().all()) ||
                   orientation.coeffs() == other.orientation.coeffs();

        return pat && met && thu && feat && mod && pos && ori;
    }
};
} // namespace opencalibration
