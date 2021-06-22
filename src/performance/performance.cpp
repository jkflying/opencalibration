#include <opencalibration/performance/performance.hpp>

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{
struct TimePoint
{
    std::string_view key;
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    enum class PointType
    {
        BEGINNING,
        ENDING
    } pointtype;

    bool operator<(const TimePoint &tc)
    {
        return timestamp < tc.timestamp;
    }
};

std::atomic<bool> _enable_counters = false;
std::vector<TimePoint> _time_points;
std::unordered_map<std::string_view, int64_t> _time_totals;
std::mutex _globals_mutex;

} // namespace

namespace opencalibration
{

void EnablePerformanceCounters(bool enable)
{
    _enable_counters.store(enable);
}

PerformanceMeasure::PerformanceMeasure(const Literal &key)
{
    initialize(key);
}

PerformanceMeasure::~PerformanceMeasure()
{
    finalize();
}

void PerformanceMeasure::reset(const opencalibration::Literal &key)
{
    finalize();
    initialize(key);
}

void PerformanceMeasure::initialize(const opencalibration::Literal &key)
{
    if (!_enable_counters.load(std::memory_order_relaxed))
    {
        return;
    }

    _key = std::string_view(key.ptr, key.len);
    _start = std::chrono::high_resolution_clock::now();
    _running = true;
}

void PerformanceMeasure::finalize()
{
    if (!_enable_counters.load(std::memory_order_relaxed))
    {
        return;
    }
    _running = false;

    auto now = std::chrono::high_resolution_clock::now();
    int64_t duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - _start).count();
    TimePoint begin{_key, _start, TimePoint::PointType::BEGINNING};
    TimePoint end{_key, now, TimePoint::PointType::ENDING};

    {
        std::lock_guard<std::mutex> lock(_globals_mutex);
        _time_totals[_key] += duration;
        _time_points.push_back(begin);
        _time_points.push_back(end);
    }
}

void ResetPerformanceCounters()
{
    std::lock_guard<std::mutex> lock(_globals_mutex);
    _time_totals.clear();
    _time_points.clear();
}

std::string TotalPerformanceSummary()
{
    using entry_t = std::pair<int64_t, std::string_view>;
    std::vector<entry_t> entries;
    std::vector<TimePoint> timeline;

    // copy from 'live' data to a vector
    {
        std::lock_guard<std::mutex> lock(_globals_mutex);
        entries.reserve(_time_totals.size());
        for (const auto &e : _time_totals)
        {
            entries.emplace_back(e.second, e.first);
        }
        timeline.reserve(_time_points.size());
        timeline.insert(timeline.begin(), _time_points.begin(), _time_points.end());
    }

    // sort totals by elapsed time
    std::sort(entries.begin(), entries.end());
    std::reverse(entries.begin(), entries.end());

    // sort timeline by timestamp
    std::sort(timeline.begin(), timeline.end());

    // iterate over timeline keeping amount of parallelism into account.
    // divide elapsed time by parallelism to get effective latency of each
    std::unordered_map<std::string_view, int64_t> wall_time_weighted;
    std::unordered_multiset<std::string_view> in_progress;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_timestamp;
    for (const auto &p : timeline)
    {
        int64_t duration = std::chrono::duration_cast<std::chrono::nanoseconds>(p.timestamp - last_timestamp).count();
        for (const auto &key : in_progress)
        {
            wall_time_weighted[key] += duration / in_progress.size();
        }
        switch (p.pointtype)
        {
        case TimePoint::PointType::BEGINNING:
            in_progress.insert(p.key);
            break;
        case TimePoint::PointType::ENDING:
            in_progress.erase(in_progress.find(p.key));
            break;
        }
        last_timestamp = p.timestamp;
    }

    // put into output
    std::ostringstream ss;
    ss << "=====================" << std::endl;
    ss << " Performance summary" << std::endl;

    ss << std::setw(25) << "Key" << std::setw(15) << "System" << std::setw(15) << "Wall" << std::setw(14)
       << "Parallelism" << std::endl;
    for (const auto &e : entries)
    {
        ss << std::setw(24) << e.second << ":";
        ss << std::fixed << std::setw(14) << std::setprecision(3) << e.first * 1e-9 << "s";
        ss << std::fixed << std::setw(14) << std::setprecision(3) << wall_time_weighted[e.second] * 1e-9 << "s";
        ss << std::fixed << std::setw(14) << std::setprecision(3) << e.first / (double)(wall_time_weighted[e.second]);
        ss << std::endl;
    }
    ss << "=====================" << std::endl;

    return ss.str();
}

} // namespace opencalibration
