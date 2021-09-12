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
    : usm::StateMachine<PipelineState, PipelineTransition>(State::INITIAL_PROCESSING), _load_stage(new LoadStage()),
      _link_stage(new LinkStage()), _relax_stage(new RelaxStage()), _step_callback([](const StepCompletionInfo &) {}),
      _batch_size(batch_size), _parallelism(parallelism == 0 ? omp_get_num_procs() : parallelism)
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

PipelineState Pipeline::chooseNextState(PipelineState currentState, Transition transition)
{
    // clang-format off
    USM_TABLE(currentState, State::COMPLETE,
        USM_STATE(transition, State::INITIAL_PROCESSING,
                  USM_MAP(Transition::NEXT, State::GLOBAL_RELAX));
        USM_STATE(transition, State::GLOBAL_RELAX,
                  USM_MAP(Transition::NEXT, State::CAMERA_PARAMETER_RELAX));
        USM_STATE(transition, State::CAMERA_PARAMETER_RELAX,
                  USM_MAP(Transition::NEXT, State::FINAL_GLOBAL_RELAX));
        USM_STATE(transition, State::FINAL_GLOBAL_RELAX,
                  USM_MAP(Transition::NEXT, State::COMPLETE));
    );
    // clang-format on
}

Pipeline::Transition Pipeline::runCurrentState(PipelineState currentState)
{
    Transition t = Transition::ERROR;
    spdlog::debug("Running {}", toString(currentState));

    switch (currentState)
    {
    case State::INITIAL_PROCESSING:
        t = initial_processing();
        break;
    case State::GLOBAL_RELAX:
        t = global_relax();
        break;
    case State::CAMERA_PARAMETER_RELAX:
        t = camera_parameter_relax();
        break;
    case State::FINAL_GLOBAL_RELAX:
        t = final_global_relax();
        break;
    case State::COMPLETE:
        t = complete();
        break;
    default:
        t = Transition::ERROR;
        break;
    }

    StepCompletionInfo info{_next_loaded_ids,    _next_linked_ids,  _next_relaxed_ids, _surfaces,
                            _graph.size_nodes(), _add_queue.size(), currentState,      stateRunCount()};
    _step_callback(info);

    return t;
}

Pipeline::Transition Pipeline::initial_processing()
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
        _relax_stage->init(_graph, _previous_linked_ids, _imageGPSLocations, false,
                           {Option::ORIENTATION, Option::GROUND_PLANE});

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

        return Transition::REPEAT;
    }
    else
    {
        return Transition::NEXT;
    }
}

Pipeline::Transition Pipeline::global_relax()
{
    _relax_stage->init(_graph, {}, _imageGPSLocations, true, {Option::ORIENTATION, Option::POINTS_3D});

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);
    _surfaces = _relax_stage->getSurfaceModels();

    if (stateRunCount() < 5)
        return Transition::REPEAT;
    else
        return Transition::NEXT;
}

Pipeline::Transition Pipeline::camera_parameter_relax()
{
    RelaxOptionSet options;
    switch (stateRunCount())
    {
    case 0:
    case 1:
        options = {Option::ORIENTATION, Option::POINTS_3D, Option::FOCAL_LENGTH};
        break;
    case 2:
        options = {Option::ORIENTATION, Option::POINTS_3D, Option::FOCAL_LENGTH, Option::LENS_DISTORTIONS_RADIAL,
                   Option::LENS_DISTORTIONS_RADIAL_BROWN2_PARAMETERIZATION};
        break;
    case 3:
        options = {Option::ORIENTATION, Option::POINTS_3D, Option::FOCAL_LENGTH, Option::LENS_DISTORTIONS_RADIAL,
                   Option::LENS_DISTORTIONS_RADIAL_BROWN24_PARAMETERIZATION};
        break;
    case 4:
        options = {Option::ORIENTATION, Option::POINTS_3D, Option::FOCAL_LENGTH, Option::LENS_DISTORTIONS_RADIAL,
                   Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION};
        break;
    default:
        options = {Option::ORIENTATION,
                   Option::POINTS_3D,
                   Option::FOCAL_LENGTH,
                   Option::LENS_DISTORTIONS_RADIAL,
                   Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION,
                   Option::LENS_DISTORTIONS_TANGENTIAL};
        break;
    }

    _relax_stage->init(_graph, {}, _imageGPSLocations, true, options);
    _relax_stage->trim_groups(1); // do it only with the largest cluster group

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);
    _surfaces = _relax_stage->getSurfaceModels();

    if (stateRunCount() < 5)
        return Transition::REPEAT;
    else
        return Transition::NEXT;
}

Pipeline::Transition Pipeline::final_global_relax()
{
    _relax_stage->init(_graph, {}, _imageGPSLocations, true, {Option::ORIENTATION, Option::POINTS_3D});

    fvec relax_funcs = _relax_stage->get_runners(_graph);
    run_parallel(relax_funcs, _parallelism);
    _next_relaxed_ids = _relax_stage->finalize(_graph);
    _surfaces = _relax_stage->getSurfaceModels();

    if (stateRunCount() < 2)
        return Transition::REPEAT;
    else
        return Transition::NEXT;
}

Pipeline::Transition Pipeline::complete()
{
    return Transition::REPEAT;
}

std::string Pipeline::toString(PipelineState state)
{
    switch (state)
    {
    case State::INITIAL_PROCESSING:
        return "Initial Processing";
    case State::GLOBAL_RELAX:
        return "Global Relax";
    case State::CAMERA_PARAMETER_RELAX:
        return "Camera Parameter Relax";
    case State::FINAL_GLOBAL_RELAX:
        return "Final Global Relax";
    case State::COMPLETE:
        return "Complete";
    };

    return "Error";
}

} // namespace opencalibration
