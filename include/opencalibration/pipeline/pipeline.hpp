#pragma once


#include <opencalibration/relax/relax.hpp>
#include <opencalibration/geo_coord/geo_coord.hpp>

#include <jk/KDTree.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class OGRCoordinateTransformation;

namespace opencalibration
{

class Pipeline
{
  public:
    enum class Status
    {
        PROCESSING,
        COMPLETE
    };

    Pipeline(size_t batch_size = 1);
    ~Pipeline();

    Status getStatus();

    void add(const std::vector<std::string> &filename);

    const MeasurementGraph& getGraph(); // warning - not threadsafe

  private:

    bool process_images(const std::vector<std::string> &paths);
    std::condition_variable _queue_condition_variable;
    std::mutex _queue_mutex;
    std::deque<std::string> _add_queue;
    bool _keep_running;

    enum class ThreadStatus
    {
        BUSY,
        IDLE
    };

    struct Runner
    {
        std::unique_ptr<std::thread> thread;
        ThreadStatus status{ThreadStatus::IDLE};
    };

    Runner _runner;

    std::mutex _graph_structure_mutex;
    MeasurementGraph _graph;

    std::mutex _kdtree_mutex;
    jk::tree::KDTree<size_t, 3> _imageGPSLocations;
    GeoCoord _coordinate_system;

};
} // namespace opencalibration
