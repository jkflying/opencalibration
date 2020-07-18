#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/extract/extract_features.hpp>

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace opencalibration
{
Pipeline::Pipeline(size_t threads)
{
    _keep_running = true;
    _runners.resize(threads);
    for (size_t i = 0; i < threads; i++)
    {
        _runners[i].first.reset(new std::thread(
            [this](size_t index) {
                std::mutex sleep_mutex;
                while (_keep_running)
                {
                    _runners[index].second = ThreadStatus::BUSY;
                    if (!process_image())
                    {
                        _runners[index].second = ThreadStatus::IDLE;
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
        runner.first->join();
    }
}

bool Pipeline::process_image()
{
    std::string path;
    {
        std::lock_guard<std::mutex> guard(_queue_mutex);
        if (_add_queue.size() > 0)
        {
            path = _add_queue.front();
            _add_queue.pop_front();
        }
        else
        {
            return false;
        }
    }

    image img;
    img.path = path;
    //     img.metadata = extract_metadata(img.path);
    img.descriptors = extract_features(img.path);

    std::lock_guard<std::mutex> img_lock(_images_mutex);
    _images.push_back(std::move(img));

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
                                      [](const auto &runner) -> bool { return runner.second == ThreadStatus::IDLE; })
               ? Status::COMPLETE
               : Status::PROCESSING;
}
} // namespace opencalibration
