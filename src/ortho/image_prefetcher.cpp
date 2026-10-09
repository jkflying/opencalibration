#include <opencalibration/ortho/image_prefetcher.hpp>

namespace opencalibration::orthomosaic
{

ImagePrefetcher::ImagePrefetcher(std::vector<ImageLoad> plan, std::function<cv::Mat(size_t image)> load,
                                 size_t num_threads)
    : plan_(std::move(plan)), load_(std::move(load))
{
    for (size_t i = 0; i < std::max<size_t>(num_threads, 1); i++)
        threads_.emplace_back([this] { run(); });
}

ImagePrefetcher::~ImagePrefetcher()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto &thread : threads_)
        thread.join();
}

cv::Mat ImagePrefetcher::get(size_t image)
{
    std::unique_lock<std::mutex> lock(mutex_);
    auto current_tile_loads_started = [&] { return next_load_ >= plan_.size() || plan_[next_load_].tile > tiles_done_; };
    cv_.wait(lock, [&] { return images_.contains(image) || (!loading_.contains(image) && current_tile_loads_started()); });
    if (auto it = images_.find(image); it != images_.end())
        return it->second;
    lock.unlock();
    return load_(image);
}

void ImagePrefetcher::finishTile()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tiles_done_++;
    }
    cv_.notify_all();
}

void ImagePrefetcher::run()
{
    std::unique_lock<std::mutex> lock(mutex_);
    while (true)
    {
        cv_.wait(lock, [&] {
            return stop_ || (next_load_ < plan_.size() && tiles_done_ >= plan_[next_load_].tiles_done_before_start);
        });
        if (stop_)
            return;
        const ImageLoad &load = plan_[next_load_++];
        images_.erase(load.evict);
        loading_.insert(load.image);
        lock.unlock();

        cv::Mat image = load_(load.image);

        lock.lock();
        loading_.erase(load.image);
        images_[load.image] = std::move(image);
        cv_.notify_all();
    }
}

} // namespace opencalibration::orthomosaic
