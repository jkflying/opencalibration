#pragma once

#include <opencalibration/types/camera_model.hpp>
#include <opencalibration/types/image_metadata.hpp>

#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace opencalibration
{

struct CameraDBEntry
{
    std::string make;
    std::string model;
    std::string lens_model;

    size_t sensor_width_px = 0;
    size_t sensor_height_px = 0;

    Eigen::Vector3d radial_distortion{0, 0, 0};
    Eigen::Vector2d tangential_distortion{0, 0};
    Eigen::Vector2d principal_point_offset{0, 0};
};

class CameraDatabase
{
  public:
    static CameraDatabase &instance();

    bool load(const std::string &path);
    bool isLoaded() const;

    std::optional<CameraDBEntry> lookup(const image_metadata::camera_info_t &camera_info) const;

  private:
    CameraDatabase() = default;
    CameraDatabase(const CameraDatabase &) = delete;
    CameraDatabase &operator=(const CameraDatabase &) = delete;

    std::vector<CameraDBEntry> _entries;
    bool _loaded = false;
    mutable std::mutex _mutex;
};

void applyDatabaseEntry(const CameraDBEntry &entry, const image_metadata::camera_info_t &camera_info,
                        CameraModel &model);

} // namespace opencalibration
