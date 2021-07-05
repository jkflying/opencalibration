#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/combinatorics/interleave.hpp>
#include <opencalibration/pipeline/link_stage.hpp>
#include <opencalibration/pipeline/load_stage.hpp>
#include <opencalibration/pipeline/relax_stage.hpp>

#include <spdlog/spdlog.h>

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace opencalibration
{
Pipeline::Pipeline(size_t parallelism)
    : _load_stage(new LoadStage()), _link_stage(new LinkStage()), _relax_stage(new RelaxStage()),
      _step_callback([](const StepCompletionInfo &) {})
{
    _keep_running = true;

    auto get_paths = [this, parallelism](std::vector<std::string> &paths) -> bool {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        while (_add_queue.size() > 0 && paths.size() < parallelism)
        {
            paths.emplace_back(std::move(_add_queue.front()));
            _add_queue.pop_front();
        }
        return paths.size() > 0;
    };

    _runner.thread.reset(new std::thread([this, get_paths]() {
        std::mutex sleep_mutex;
        std::vector<std::string> paths;
        std::vector<size_t> previous_loaded_ids, previous_linked_ids, next_loaded_ids, next_linked_ids,
            next_relaxed_ids;
        size_t empty_iterations = 0;
        while (_keep_running)
        {
            paths.clear();
            if (get_paths(paths) || empty_iterations++ < 5)
            {
                _runner.status = ThreadStatus::BUSY;
                next_loaded_ids.clear();
                next_linked_ids.clear();

                if (previous_loaded_ids.size() > 0 || previous_linked_ids.size() > 0)
                {
                    empty_iterations = 0;
                }

                process_images(paths, previous_loaded_ids, previous_linked_ids, next_loaded_ids, next_linked_ids,
                               next_relaxed_ids);

                StepCompletionInfo info{next_loaded_ids, next_linked_ids, next_relaxed_ids, _graph.size_nodes(),
                                        _add_queue.size()};
                _step_callback(info);

                std::swap(previous_linked_ids, next_linked_ids);
                std::swap(previous_loaded_ids, next_loaded_ids);
            }
            else
            {
                _runner.status = ThreadStatus::IDLE;
                std::unique_lock<std::mutex> lck(sleep_mutex);
                _queue_condition_variable.wait_for(lck, 1s);
            }
        }
    }));
}

Pipeline::~Pipeline()
{
    _keep_running = false;
    _queue_condition_variable.notify_all();
    if (_runner.thread != nullptr)
    {
        _runner.thread->join();
    }
}

void Pipeline::process_images(const std::vector<std::string> &paths_to_load,
                              const std::vector<size_t> &previous_loaded_ids,
                              const std::vector<size_t> &previous_linked_ids, std::vector<size_t> &next_loaded_ids,
                              std::vector<size_t> &next_linked_ids, std::vector<size_t> &next_relaxed_ids)
{

    using fvec = std::vector<std::function<void()>>;

    bool last_iteration = paths_to_load.size() + previous_loaded_ids.size() == 0;
    _load_stage->init(paths_to_load);
    _link_stage->init(_graph, _imageGPSLocations, previous_loaded_ids);
    _relax_stage->init(_graph, previous_linked_ids, _imageGPSLocations, last_iteration);

    fvec funcs;
    {
        fvec load_funcs = _load_stage->get_runners();
        fvec link_funcs = _link_stage->get_runners(_graph);
        fvec relax_funcs = _relax_stage->get_runners(_graph);
        funcs = interleave<fvec>({load_funcs, link_funcs, relax_funcs});
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < (int)funcs.size(); i++)
    {
        funcs[i]();
    }

    next_loaded_ids = _load_stage->finalize(_coordinate_system, _graph, _imageGPSLocations);
    next_linked_ids = _link_stage->finalize(_graph);
    next_relaxed_ids = _relax_stage->finalize(_graph);
}

void Pipeline::add(const std::vector<std::string> &paths)
{

    std::lock_guard<std::mutex> guard(_queue_mutex);
    _add_queue.insert(_add_queue.end(), paths.begin(), paths.end());
    _queue_condition_variable.notify_all();
}

Pipeline::Status Pipeline::getStatus()
{
    bool queue_empty;
    {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        queue_empty = _add_queue.size() == 0;
    }
    return queue_empty && _runner.status == ThreadStatus::IDLE ? Status::COMPLETE : Status::PROCESSING;
}

const MeasurementGraph &Pipeline::getGraph()
{
    return _graph;
}
} // namespace opencalibration
