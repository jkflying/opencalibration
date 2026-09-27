#include <opencalibration/ortho/image_cache.hpp>

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
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
    std::unique_lock<std::mutex> lock(cache_mutex_);

    // Wait if another thread is loading this image
    cv_.wait(lock, [this, node_id] { return loading_.count(node_id) == 0; });

    // Check if image is now cached
    auto it = cache_.find(node_id);
    if (it != cache_.end())
    {
        it->second.last_access_time = access_counter_++;
        cache_hits_++;
        return it->second.image;
    }

    cache_misses_++;
    return loadAndInsert(lock, node_id, path);
}

bool FullResolutionImageCache::tryPrefetch(size_t node_id, const std::string &path)
{
    std::unique_lock<std::mutex> lock(cache_mutex_);

    if (cache_.count(node_id) > 0 || loading_.count(node_id) > 0)
        return true;

    if (cache_.size() >= max_cache_size_)
    {
        if (!next_use_)
            return false;
        size_t furthest = 0;
        for (const auto &entry : cache_)
            furthest = std::max(furthest, next_use_(entry.first));
        if (next_use_(node_id) >= furthest)
            return false;
    }

    loadAndInsert(lock, node_id, path);
    return true;
}

cv::Mat FullResolutionImageCache::loadAndInsert(std::unique_lock<std::mutex> &lock, size_t node_id,
                                                const std::string &path)
{
    loading_.insert(node_id);
    lock.unlock();

    cv::Mat image = cv::imread(path);

    lock.lock();

    loading_.erase(node_id);

    if (image.empty())
    {
        spdlog::warn("Failed to load image: {}", path);
        cv_.notify_all();
        return cv::Mat();
    }

    if (cache_.size() >= max_cache_size_)
    {
        auto evict_before = [this](const auto &a, const auto &b) {
            if (next_use_)
                return next_use_(a.first) > next_use_(b.first);
            return a.second.last_access_time < b.second.last_access_time;
        };
        auto victim = std::min_element(cache_.begin(), cache_.end(), evict_before);
        spdlog::debug("Evicted image {} from cache", victim->first);
        cache_.erase(victim);
    }

    CachedImage cached{image, access_counter_++};
    cache_[node_id] = cached;

    spdlog::debug("Loaded image {} into cache (cache size: {}/{})", node_id, cache_.size(), max_cache_size_);

    cv_.notify_all();

    return image;
}

void FullResolutionImageCache::setNextUse(std::function<size_t(size_t node_id)> next_use)
{
    std::lock_guard<std::mutex> lock(cache_mutex_);
    next_use_ = std::move(next_use);
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
