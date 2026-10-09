#pragma once

#include <opencalibration/tile_ordering/tile_ordering.hpp>

#include <ankerl/unordered_dense.h>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <opencv2/core.hpp>
#include <thread>
#include <vector>

namespace opencalibration::orthomosaic
{

class ImagePrefetcher
{
  public:
    ImagePrefetcher(std::vector<ImageLoad> plan, std::function<cv::Mat(size_t image)> load, size_t num_threads);
    ~ImagePrefetcher();
    ImagePrefetcher(const ImagePrefetcher &) = delete;
    ImagePrefetcher &operator=(const ImagePrefetcher &) = delete;

    cv::Mat get(size_t image);
    void finishTile();

  private:
    void run();

    const std::vector<ImageLoad> plan_;
    const std::function<cv::Mat(size_t)> load_;

    std::mutex mutex_;
    std::condition_variable cv_;
    ankerl::unordered_dense::map<size_t, cv::Mat> images_;
    ankerl::unordered_dense::set<size_t> loading_;
    size_t next_load_ = 0;
    size_t tiles_done_ = 0;
    bool stop_ = false;
    std::vector<std::thread> threads_;
};

} // namespace opencalibration::orthomosaic
