#pragma once

#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/types/measurement_graph.hpp>

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
class LoadStage;
class LinkStage;
class RelaxStage;

class Pipeline
{
  public:
    enum class Status
    {
        PROCESSING,
        COMPLETE
    };

    struct StepCompletionInfo
    {
        std::reference_wrapper<std::vector<size_t>> loaded_ids, linked_ids, relaxed_ids;
        size_t images_loaded, queue_size_remaining;
    };
    using StepCompletionCallback = std::function<void(const StepCompletionInfo &)>;

    Pipeline(size_t batch_size = 1);
    ~Pipeline();

    Status getStatus();

    void add(const std::vector<std::string> &filename);

    // warning - not threadsafe, only access from callback or when finished
    const MeasurementGraph &getGraph();

    const GeoCoord &getCoord() const
    {
        return _coordinate_system;
    }

    void set_callback(const StepCompletionCallback &step_complete)
    {
        _step_callback = step_complete;
    }

  private:
    std::unique_ptr<LoadStage> _load_stage;
    std::unique_ptr<LinkStage> _link_stage;
    std::unique_ptr<RelaxStage> _relax_stage;

    void process_images(const std::vector<std::string> &paths_to_load, const std::vector<size_t> &previous_loaded_ids,
                        const std::vector<size_t> &previous_linked_ids, std::vector<size_t> &next_loaded_ids,
                        std::vector<size_t> &next_linked_ids, std::vector<size_t> &next_relaxed_ids);

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
    jk::tree::KDTree<size_t, 3> _imageGPSLocations;
    GeoCoord _coordinate_system;

    StepCompletionCallback _step_callback;
};
} // namespace opencalibration
