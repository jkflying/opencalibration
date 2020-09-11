#pragma once


#include <Eigen/Core>

#include <string>
#include <memory>

class OGRCoordinateTransformation;

namespace opencalibration
{
class GeoCoord
{
  public:
    GeoCoord();
    ~GeoCoord();

    bool isInitialized() const;

    bool setOrigin(double latitude, double longitude);

    Eigen::Vector3d toLocalCS(double latitude, double longitude, double altitude) const;

    Eigen::Vector3d toWGS84(const Eigen::Vector3d& local) const;

    std::string getWKT() const;

  private:
    bool _initialized = false;

    std::unique_ptr<OGRCoordinateTransformation> _to_local, _to_wgs84;
};
} // namespace opencalibration
