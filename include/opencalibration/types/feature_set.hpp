#pragma once

#include <opencalibration/types/feature_2d.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <utility>
#include <vector>

namespace opencalibration
{

class FeatureSet
{
  public:
    using Vec = std::vector<feature_2d>;
    using Loader = std::function<Vec()>;

    FeatureSet() = default;
    FeatureSet(Vec features) : _size(features.size()), _resident(std::move(features))
    {
    }

    static FeatureSet stored(size_t size, Loader loader)
    {
        FeatureSet set;
        set._size = size;
        set._loader = std::move(loader);
        return set;
    }

    [[nodiscard]] size_t size() const
    {
        return _size;
    }
    [[nodiscard]] bool empty() const
    {
        return _size == 0;
    }
    [[nodiscard]] bool isStored() const
    {
        return static_cast<bool>(_loader);
    }

    [[nodiscard]] Vec load() const &
    {
        if (!_loader)
            return _resident;
        const auto start = std::chrono::steady_clock::now();
        Vec features = _loader();
        loadStats().record(features.size(), start);
        return features;
    }
    [[nodiscard]] Vec load() &&
    {
        return _loader ? std::as_const(*this).load() : std::move(_resident);
    }

    bool operator==(const FeatureSet &other) const
    {
        return size() == other.size() && load() == other.load();
    }

    struct LoadStats
    {
        std::atomic<size_t> loads{0}, features{0}, nanoseconds{0}, lock_wait_nanoseconds{0};
        void record(size_t count, std::chrono::steady_clock::time_point start)
        {
            loads += 1;
            features += count;
            nanoseconds +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        }
    };
    static LoadStats &loadStats()
    {
        static LoadStats s;
        return s;
    }

  private:
    size_t _size = 0;
    Vec _resident;
    Loader _loader;
};

} // namespace opencalibration
