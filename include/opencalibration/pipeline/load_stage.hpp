#pragma once

#include <jk/KDTree.h>
#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/types/image.hpp>
#include <opencalibration/types/measurement_graph.hpp>

#include <mutex>
#include <vector>

namespace opencalibration
{

class LoadStage
{
  public:
    void init(const std::vector<std::string> &paths_to_load);
    std::vector<std::function<void()>> get_runners();
    std::vector<size_t> finalize(GeoCoord &coordinate_system, MeasurementGraph &graph,
                                 jk::tree::KDTree<size_t, 3> &imageGPSLocations);

  private:
    std::mutex _images_mutex;
    std::vector<std::pair<size_t, image>> _images;

    std::vector<std::string> _paths_to_load;
};
} // namespace opencalibration
