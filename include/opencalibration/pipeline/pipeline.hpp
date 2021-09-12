#pragma once

#include <opencalibration/geo_coord/geo_coord.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/surface_model.hpp>

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
    CAMERA_PARAMETER_RELAX,
    FINAL_GLOBAL_RELAX,
    COMPLETE
};

enum class PipelineTransition
{
    REPEAT,
    NEXT,
    ERROR
};

class Pipeline : public usm::StateMachine<PipelineState, PipelineTransition>
{
  public:
    struct StepCompletionInfo
    {
        std::reference_wrapper<std::vector<size_t>> loaded_ids, linked_ids;
        std::reference_wrapper<std::vector<std::vector<size_t>>> relaxed_ids;
        std::reference_wrapper<std::vector<surface_model>> surfaces;
        size_t images_loaded, queue_size_remaining;
        PipelineState state;
        uint64_t state_iteration;
    };
    using StepCompletionCallback = std::function<void(const StepCompletionInfo &)>;

    Pipeline(size_t batch_size = 1, size_t parallelism = 0); // parallelism = 0 --> unlimited
    ~Pipeline();

    void add(const std::vector<std::string> &filename);

    // warning - not threadsafe, only access from callback or when finished
    const MeasurementGraph &getGraph()
    {
        return _graph;
    }

    const GeoCoord &getCoord() const
    {
        return _coordinate_system;
    }

    const std::vector<surface_model> &getSurfaces()
    {
        return _surfaces;
    }

    void set_callback(const StepCompletionCallback &step_complete)
    {
        _step_callback = step_complete;
    }

    static std::string toString(PipelineState state);

  protected:
    PipelineState chooseNextState(PipelineState currentState, PipelineTransition transition) override;
    PipelineTransition runCurrentState(PipelineState currentState) override;

  private:
    PipelineTransition initial_processing();
    PipelineTransition global_relax();
    PipelineTransition camera_parameter_relax();
    PipelineTransition final_global_relax();
    PipelineTransition complete();

    std::vector<size_t> _previous_loaded_ids, _previous_linked_ids, _next_loaded_ids, _next_linked_ids;

    std::vector<std::vector<size_t>> _next_relaxed_ids;

    std::unique_ptr<LoadStage> _load_stage;
    std::unique_ptr<LinkStage> _link_stage;
    std::unique_ptr<RelaxStage> _relax_stage;

    std::condition_variable _queue_condition_variable;
    std::mutex _queue_mutex;
    std::deque<std::string> _add_queue;

    MeasurementGraph _graph;
    std::vector<surface_model> _surfaces;
    jk::tree::KDTree<size_t, 3> _imageGPSLocations;
    GeoCoord _coordinate_system;

    StepCompletionCallback _step_callback;

    size_t _batch_size;
    size_t _parallelism;
};
} // namespace opencalibration
