#pragma once


#include <opencalibration/relax/graph.hpp>
#include <opencalibration/types/camera_relations.hpp>
#include <opencalibration/types/image.hpp>

#include <jk/KDTree.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

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

    Pipeline(size_t threads = 1);
    ~Pipeline();

    Status getStatus();

    void add(const std::string &filename);

  private:

    bool process_image(const std::string &filename);
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

    std::vector<Runner> _runners;

    std::mutex _graph_structure_mutex;
    DirectedGraph<image, camera_relations> _graph;

    std::mutex _kdtree_mutex;
    jk::tree::KDTree<size_t, 3> _imageGPSLocations;
};
} // namespace opencalibration
