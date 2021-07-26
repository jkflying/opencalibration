#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/combinatorics/interleave.hpp>
#include <opencalibration/pipeline/link_stage.hpp>
#include <opencalibration/pipeline/load_stage.hpp>
#include <opencalibration/pipeline/relax_stage.hpp>

#include <spdlog/spdlog.h>

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;
using fvec = std::vector<std::function<void()>>;

namespace
{

void run_parallel(fvec &funcs, int parallelism)
{
#pragma omp parallel for schedule(dynamic, 1) num_threads(parallelism)
    for (int i = 0; i < (int)funcs.size(); i++)
    {
        funcs[i]();
    }
}

} // namespace

namespace opencalibration
{
Pipeline::Pipeline(size_t parallelism)
    : usm::StateMachine<PipelineState>(PipelineState::INITIAL_PROCESSING), _load_stage(new LoadStage()),
      _link_stage(new LinkStage()), _relax_stage(new RelaxStage()), _step_callback([](const StepCompletionInfo &) {}),
      _parallelism(parallelism)
{
}

Pipeline::~Pipeline()
{
}

void Pipeline::add(const std::vector<std::string> &paths)
{

    std::lock_guard<std::mutex> guard(_queue_mutex);
    _add_queue.insert(_add_queue.end(), paths.begin(), paths.end());
    _queue_condition_variable.notify_all();
}

const MeasurementGraph &Pipeline::getGraph()
{
    return _graph;
}

PipelineState Pipeline::chooseNextState(opencalibration::PipelineState currentState, usm::Transition transition)
{
    using namespace usm;
    using PS = PipelineState;

    // clang-format off
    USM_TABLE(currentState, PS::COMPLETE,
        USM_STATE(transition, PS::INITIAL_PROCESSING,   USM_MAP(NEXT1, PS::GLOBAL_RELAX););
        USM_STATE(transition, PS::GLOBAL_RELAX,         USM_MAP(NEXT1, PS::CAMERA_PARAMETERS););
        USM_STATE(transition, PS::CAMERA_PARAMETERS,    USM_MAP(NEXT1, PS::FOCAL_RELAX););
        USM_STATE(transition, PS::FOCAL_RELAX,       USM_MAP(NEXT1, PS::COMPLETE););
    );
    // clang-format on
}

usm::Transition Pipeline::runCurrentState(opencalibration::PipelineState currentState)
{
    using PS = PipelineState;

    usm::Transition t = usm::ERROR;
    spdlog::debug("Running {}", toString(currentState));

    switch (currentState)
    {
    case PS::INITIAL_PROCESSING:
        t = initial_processing();
        break;
    case PS::GLOBAL_RELAX:
        t = global_relax();
        break;
    case PS::CAMERA_PARAMETERS:
        t = usm::NEXT1;
        break;
    case PS::FOCAL_RELAX:
        t = focal_relax();
        break;
    case PS::COMPLETE:
        t = complete();
        break;
    default:
        t = usm::ERROR;
        break;
    }

    StepCompletionInfo info{_next_loaded_ids,  _next_linked_ids, _next_relaxed_ids, _graph.size_nodes(),
                            _add_queue.size(), currentState,     stateRunCount()};
    _step_callback(info);

    return t;
}

usm::Transition Pipeline::initial_processing()
{
    auto get_paths = [this](std::vector<std::string> &paths) -> bool {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        while (_add_queue.size() > 0 && paths.size() < _parallelism)
        {
            paths.emplace_back(std::move(_add_queue.front()));
            _add_queue.pop_front();
        }
        return paths.size() > 0;
    };

    std::vector<std::string> paths;
    if (get_paths(paths) || _next_loaded_ids.size() > 0 || _next_linked_ids.size() > 0)
    {
        _previous_loaded_ids = std::move(_next_loaded_ids);
        _next_loaded_ids.clear();

        _previous_linked_ids = std::move(_next_linked_ids);
        _next_linked_ids.clear();

        _load_stage->init(_graph, paths);
        _link_stage->init(_graph, _imageGPSLocations, _previous_loaded_ids);
        _relax_stage->init(_graph, _previous_linked_ids, _imageGPSLocations, false, {Option::ORIENTATION});

        fvec funcs;
        {
            fvec load_funcs = _load_stage->get_runners();
            fvec link_funcs = _link_stage->get_runners(_graph);
            fvec relax_funcs = _relax_stage->get_runners(_graph);
            funcs = interleave<fvec>({load_funcs, link_funcs, relax_funcs});
        }

        run_parallel(funcs, _parallelism);

        _next_loaded_ids = _load_stage->finalize(_coordinate_system, _graph, _imageGPSLocations);
        _next_linked_ids = _link_stage->finalize(_graph);
        _next_relaxed_ids = _relax_stage->finalize(_graph);

        return usm::Transition::REPEAT;
    }
    else
    {
        return usm::Transition::NEXT1;
    }
}

usm::Transition Pipeline::global_relax()
{
    _relax_stage->init(_graph, {}, _imageGPSLocations, true, {Option::ORIENTATION, Option::POINTS_3D});

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);

    if (stateRunCount() < 5)
        return usm::Transition::REPEAT;
    else
        return usm::Transition::NEXT1;
}

usm::Transition Pipeline::focal_relax()
{
    _relax_stage->init(_graph, {}, _imageGPSLocations, true,
                       {Option::ORIENTATION, Option::POINTS_3D, Option::FOCAL_LENGTH});
    _relax_stage->trim_groups(1); // do it only with the largest cluster group

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);

    if (stateRunCount() < 5)
        return usm::Transition::REPEAT;
    else
        return usm::Transition::NEXT1;
}

usm::Transition Pipeline::complete()
{
    return usm::Transition::REPEAT;
}

std::string Pipeline::toString(PipelineState state)
{
    using PS = PipelineState;
    switch (state)
    {
    case PS::INITIAL_PROCESSING:
        return "Initial Processing";
    case PS::GLOBAL_RELAX:
        return "Global Relax";
    case PS::CAMERA_PARAMETERS:
        return "Camera Parameters";
    case PS::FOCAL_RELAX:
        return "Focal Length Relax";
    case PS::COMPLETE:
        return "Complete";
    };

    return "Error";
}

} // namespace opencalibration
