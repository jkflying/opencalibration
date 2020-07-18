#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/extract/extract_features.hpp>
#include <opencalibration/extract/extract_metadata.hpp>

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace opencalibration
{
Pipeline::Pipeline(size_t threads)
{
    _keep_running = true;
    auto get_path = [this](std::string &path, ThreadStatus &status) -> bool {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        if (_add_queue.size() > 0)
        {
            path = _add_queue.front();
            _add_queue.pop_front();
            status = ThreadStatus::BUSY;
            return true;
        }
        else
        {
            status = ThreadStatus::IDLE;
            return false;
        }
    };
    _runners.resize(threads);
    for (size_t i = 0; i < threads; i++)
    {
        _runners[i].thread.reset(new std::thread(
            [this, get_path](size_t index) {
                std::mutex sleep_mutex;
                while (_keep_running)
                {
                    std::string path;
                    if (get_path(path, _runners[index].status))
                    {
                        process_image(path);
                    }
                    else
                    {
                        std::unique_lock<std::mutex> lck(sleep_mutex);
                        _queue_condition_variable.wait_for(lck, 1s);
                    }
                }
            },
            i));
    }
}

Pipeline::~Pipeline()
{
    _keep_running = false;
    _queue_condition_variable.notify_all();
    for (auto &runner : _runners)
    {
        runner.thread->join();
    }
}

bool Pipeline::process_image(const std::string &path)
{

    image img;
    img.path = path;
    img.metadata = extract_metadata(img.path);
    img.descriptors = extract_features(img.path);

    // find N nearest

    // match

    // ransac

    // add correspondences

    std::lock_guard<std::mutex> img_lock(_graph_structure_mutex);
    size_t node_id = _graph.addNode(std::move(img));

    return true;
}

void Pipeline::add(const std::string &path)
{
    {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        _add_queue.push_back(path);
        _queue_condition_variable.notify_one();
    }
}

Pipeline::Status Pipeline::getStatus()
{
    bool queue_empty;
    {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        queue_empty = _add_queue.size() == 0;
    }
    return queue_empty && std::all_of(_runners.begin(), _runners.end(),
                                      [](const Runner &runner) -> bool { return runner.status == ThreadStatus::IDLE; })
               ? Status::COMPLETE
               : Status::PROCESSING;
}
} // namespace opencalibration
