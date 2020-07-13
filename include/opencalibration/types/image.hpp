#pragma once

#include <opencalibration/types/feature_2d.hpp>

#include <memory>
#include <vector>

namespace opencalibration
{

struct image_metadata
{
};

struct image_pixeldata
{
    enum Format {


    } format;
    std::vector<unsigned char> data;
};

struct image
{

    std::string path;

    std::unique_ptr<image_metadata> metadata;
    std::unique_ptr<image_pixeldata> pixeldata;

    std::vector<feature_2d> descriptors;
};
} // namespace opencalibration
