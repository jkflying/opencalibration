#pragma once

#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/types/measurement_graph.hpp>

#include <jk/KDTree.h>
#include <usm.hpp>

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

enum class PipelineState
{
    INITIAL_PROCESSING,
    GLOBAL_RELAX,
    CAMERA_PARAMETERS,
    FOCAL_RELAX,
    COMPLETE
};

class Pipeline : public usm::StateMachine<PipelineState>
{
  public:
    struct StepCompletionInfo
    {
        std::reference_wrapper<std::vector<size_t>> loaded_ids, linked_ids;
        std::reference_wrapper<std::vector<std::vector<size_t>>> relaxed_ids;
        size_t images_loaded, queue_size_remaining;
        PipelineState state;
        uint64_t state_iteration;
    };
    using StepCompletionCallback = std::function<void(const StepCompletionInfo &)>;

    Pipeline(size_t parallelism = 1);
    ~Pipeline();

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

    static std::string toString(PipelineState state);

  protected:
    PipelineState chooseNextState(opencalibration::PipelineState currentState, usm::Transition transition) override;
    usm::Transition runCurrentState(opencalibration::PipelineState currentState) override;

  private:
    usm::Transition initial_processing();
    usm::Transition global_relax();
    usm::Transition focal_relax();
    usm::Transition complete();

    std::vector<size_t> _previous_loaded_ids, _previous_linked_ids, _next_loaded_ids, _next_linked_ids;

    std::vector<std::vector<size_t>> _next_relaxed_ids;

    std::unique_ptr<LoadStage> _load_stage;
    std::unique_ptr<LinkStage> _link_stage;
    std::unique_ptr<RelaxStage> _relax_stage;

    std::condition_variable _queue_condition_variable;
    std::mutex _queue_mutex;
    std::deque<std::string> _add_queue;

    MeasurementGraph _graph;
    jk::tree::KDTree<size_t, 3> _imageGPSLocations;
    GeoCoord _coordinate_system;

    StepCompletionCallback _step_callback;

    size_t _parallelism;
};
} // namespace opencalibration
