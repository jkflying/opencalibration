#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/pipeline/link_stage.hpp>
#include <opencalibration/pipeline/load_stage.hpp>

#include <spdlog/spdlog.h>

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace opencalibration
{
Pipeline::Pipeline(size_t batch_size)
{
    _keep_running = true;

    auto get_paths = [this, batch_size](std::vector<std::string> &paths) -> bool {
        while (_add_queue.size() > 0 && paths.size() < batch_size)
        {
            paths.emplace_back(std::move(_add_queue.front()));
            _add_queue.pop_front();
        }
        return paths.size() > 0;
    };

    _runner.thread.reset(new std::thread([this, batch_size, get_paths]() {
        std::mutex sleep_mutex;
        std::vector<std::string> paths;
        std::vector<size_t> loaded_ids, next_ids;
        paths.reserve(batch_size);
        loaded_ids.reserve(batch_size);
        next_ids.reserve(batch_size);
        while (_keep_running)
        {
            paths.clear();
            next_ids.clear();
            if (get_paths(paths) || loaded_ids.size() > 0)
            {
                _runner.status = ThreadStatus::BUSY;
                process_images(loaded_ids, paths, next_ids);
                std::swap(loaded_ids, next_ids);
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

void Pipeline::process_images(const std::vector<size_t> &loaded_ids, const std::vector<std::string> &paths_to_load,
                              std::vector<size_t> &next_ids)
{
    std::vector<std::function<void()>> funcs;

    LoadStage loadStage;
    loadStage.init(paths_to_load);
    std::vector<std::function<void()>> load_funcs = loadStage.get_runners();

    BuildLinksStage linkStage;
    linkStage.init(_graph, _imageGPSLocations, loaded_ids);
    std::vector<std::function<void()>> link_funcs = linkStage.get_runners(_graph);

    funcs.reserve(load_funcs.size() + link_funcs.size());
    // interleave the functions to spread resource usage across the excution
    while (load_funcs.size() > 0 || link_funcs.size() > 0)
    {
        if (load_funcs.size() > 0)
        {
            funcs.push_back(std::move(load_funcs.back()));
            load_funcs.pop_back();
        }
        if (link_funcs.size() > 0)
        {
            funcs.push_back(std::move(link_funcs.back()));
            link_funcs.pop_back();
        }
    }
    std::reverse(funcs.begin(), funcs.end());

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)funcs.size(); i++)
    {
        funcs[i]();
    }

    std::lock_guard<std::mutex> graph_lock(_graph_structure_mutex);
    next_ids = loadStage.finalize(_coordinate_system, _graph, _imageGPSLocations);
    linkStage.finalize(_graph);

    initializeOrientation(loaded_ids, _graph);
    // TODO:
    // relaxSubset(node_ids, _graph, _graph_structure_mutex);
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
