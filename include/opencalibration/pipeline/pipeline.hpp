#pragma once

#include <opencalibration/types/image.hpp>

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace opencalibration
{

class Pipeline
{
  public:
    enum class Status
    {
        PROCESSING,
        COMPLETE
    };

    Pipeline(size_t threads = 1);
    ~Pipeline();

    Status getStatus();

    void add(const std::string &filename);
    bool process_image();

  private:

    std::condition_variable _queue_condition_variable;
    std::mutex _queue_mutex;
    std::deque<std::string> _add_queue;
    bool _keep_running;

    enum class ThreadStatus
    {
        BUSY,
        IDLE
    };

    std::vector<std::pair<std::unique_ptr<std::thread>,ThreadStatus>> _runners;

    std::mutex _images_mutex;
    std::vector<image> _images;
};
} // namespace opencalibration
