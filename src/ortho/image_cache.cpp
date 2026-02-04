#include <opencalibration/ortho/image_cache.hpp>

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <thread>

namespace opencalibration
{
namespace orthomosaic
{

FullResolutionImageCache::FullResolutionImageCache(size_t max_size) : max_cache_size_(max_size)
{
}

cv::Mat FullResolutionImageCache::getImage(size_t node_id, const std::string &path)
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(cache_mutex_);

        auto it = cache_.find(node_id);
        if (it != cache_.end())
        {
            it->second.last_access_time = access_counter_++;
            cache_hits_++;
            return it->second.image;
        }

        if (loading_.count(node_id) == 0)
        {
            loading_.insert(node_id);
            cache_misses_++;
            break;
        }

        lock.unlock();
        std::this_thread::yield();
    }

    if (cv::getNumThreads() != 1)
    {
        cv::setNumThreads(1);
    }

    cv::Mat image = cv::imread(path);

    std::lock_guard<std::mutex> lock(cache_mutex_);

    loading_.erase(node_id);

    if (image.empty())
    {
        spdlog::warn("Failed to load image: {}", path);
        return cv::Mat();
    }

    if (cache_.size() >= max_cache_size_)
    {
        size_t oldest_node_id = 0;
        size_t oldest_access_time = SIZE_MAX;

        for (const auto &entry : cache_)
        {
            if (entry.second.last_access_time < oldest_access_time)
            {
                oldest_access_time = entry.second.last_access_time;
                oldest_node_id = entry.first;
            }
        }

        cache_.erase(oldest_node_id);
        spdlog::debug("Evicted image {} from cache", oldest_node_id);
    }

    CachedImage cached{image, access_counter_++};
    cache_[node_id] = cached;

    spdlog::debug("Loaded image {} into cache (cache size: {}/{})", node_id, cache_.size(), max_cache_size_);

    return image;
}

void FullResolutionImageCache::clear()
{
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    spdlog::debug("Cleared image cache. Stats: {} hits, {} misses", cache_hits_, cache_misses_);
}

size_t FullResolutionImageCache::getCacheHits() const
{
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_hits_;
}

size_t FullResolutionImageCache::getCacheMisses() const
{
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_misses_;
}

} // namespace orthomosaic
} // namespace opencalibration
