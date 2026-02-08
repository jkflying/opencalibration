#pragma once

#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace opencalibration
{
namespace orthomosaic
{
class FullResolutionImageCache
{
  private:
    struct CachedImage
    {
        cv::Mat image;
        size_t last_access_time;
    };

    std::unordered_map<size_t, CachedImage> cache_;
    std::unordered_set<size_t> loading_;
    size_t max_cache_size_;
    size_t access_counter_ = 0;
    mutable std::mutex cache_mutex_;
    mutable std::condition_variable cv_;
    size_t cache_hits_ = 0;
    size_t cache_misses_ = 0;

  public:
    explicit FullResolutionImageCache(size_t max_size = 10);

    // Load image from path (uses cache if available)
    cv::Mat getImage(size_t node_id, const std::string &path);

    // Clear entire cache (between tiles)
    void clear();

    // Get cache statistics
    size_t getCacheHits() const;
    size_t getCacheMisses() const;
};

} // namespace orthomosaic
} // namespace opencalibration
