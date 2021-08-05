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
Pipeline::Pipeline(size_t batch_size, size_t parallelism)
    : usm::StateMachine<PipelineState, PipelineTransition>(PipelineState::INITIAL_PROCESSING),
      _load_stage(new LoadStage()), _link_stage(new LinkStage()), _relax_stage(new RelaxStage()),
      _step_callback([](const StepCompletionInfo &) {}), _batch_size(batch_size),
      _parallelism(parallelism == 0 ? omp_get_num_procs() : parallelism)
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

PipelineState Pipeline::chooseNextState(PipelineState currentState, PipelineTransition transition)
{
    using namespace usm;
    using PS = PipelineState;

    // clang-format off
    USM_TABLE(currentState, PS::COMPLETE,
        USM_STATE(transition, PS::INITIAL_PROCESSING,
                  USM_MAP(PipelineTransition::NEXT, PS::GLOBAL_RELAX));
        USM_STATE(transition, PS::GLOBAL_RELAX,
                  USM_MAP(PipelineTransition::NEXT, PS::FOCAL_RELAX));
        USM_STATE(transition, PS::FOCAL_RELAX,
                  USM_MAP(PipelineTransition::NEXT, PS::CAMERA_PARAMETERS));
        USM_STATE(transition, PS::CAMERA_PARAMETERS,
                  USM_MAP(PipelineTransition::NEXT, PS::FINAL_GLOBAL_RELAX));
        USM_STATE(transition, PS::FINAL_GLOBAL_RELAX,
                  USM_MAP(PipelineTransition::NEXT, PS::COMPLETE));
    );
    // clang-format on
}

PipelineTransition Pipeline::runCurrentState(PipelineState currentState)
{
    using PS = PipelineState;

    PipelineTransition t = PipelineTransition::ERROR;
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
        t = PipelineTransition::NEXT;
        _next_relaxed_ids.clear();
        break;
    case PS::FOCAL_RELAX:
        t = focal_relax();
        break;
    case PS::FINAL_GLOBAL_RELAX:
        t = final_global_relax();
        break;
    case PS::COMPLETE:
        t = complete();
        break;
    default:
        t = PipelineTransition::ERROR;
        break;
    }

    StepCompletionInfo info{_next_loaded_ids,  _next_linked_ids, _next_relaxed_ids, _graph.size_nodes(),
                            _add_queue.size(), currentState,     stateRunCount()};
    _step_callback(info);

    return t;
}

PipelineTransition Pipeline::initial_processing()
{
    auto get_paths = [this](std::vector<std::string> &paths) -> bool {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        while (_add_queue.size() > 0 && paths.size() < _batch_size)
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

        return PipelineTransition::REPEAT;
    }
    else
    {
        return PipelineTransition::NEXT;
    }
}

PipelineTransition Pipeline::global_relax()
{
    _relax_stage->init(_graph, {}, _imageGPSLocations, true, {Option::ORIENTATION, Option::POINTS_3D});

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);

    if (stateRunCount() < 5)
        return PipelineTransition::REPEAT;
    else
        return PipelineTransition::NEXT;
}

PipelineTransition Pipeline::focal_relax()
{
    _relax_stage->init(_graph, {}, _imageGPSLocations, true,
                       {Option::ORIENTATION, Option::POINTS_3D, Option::FOCAL_LENGTH});
    _relax_stage->trim_groups(1); // do it only with the largest cluster group

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);

    if (stateRunCount() < 2)
        return PipelineTransition::REPEAT;
    else
        return PipelineTransition::NEXT;
}

PipelineTransition Pipeline::final_global_relax()
{
    _relax_stage->init(_graph, {}, _imageGPSLocations, true, {Option::ORIENTATION, Option::POINTS_3D});

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);

    if (stateRunCount() < 2)
        return PipelineTransition::REPEAT;
    else
        return PipelineTransition::NEXT;
}

PipelineTransition Pipeline::complete()
{
    return PipelineTransition::REPEAT;
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
    case PS::FINAL_GLOBAL_RELAX:
        return "Final Global Relax";
    case PS::COMPLETE:
        return "Complete";
    };

    return "Error";
}

} // namespace opencalibration
