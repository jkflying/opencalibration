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

    bool isInitialized();

    bool setOrigin(double latitude, double longitude);

    Eigen::Vector3d toLocalCS(double latitude, double longitude, double altitude);

    std::string getWKT();

  private:
    bool _initialized = false;

    std::unique_ptr<OGRCoordinateTransformation> _transform;
};
} // namespace opencalibration
