#include <opencalibration/performance/performance.hpp>

#include <algorithm>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace
{
std::unordered_map<std::string_view, int64_t> _time_totals;
std::mutex _time_totals_mutex;

} // namespace

namespace opencalibration
{
PerformanceMeasure::PerformanceMeasure(const Literal &key)
    : _key(key.ptr, key.len), _start(std::chrono::high_resolution_clock::now())
{
}

PerformanceMeasure::~PerformanceMeasure()
{
    auto now = std::chrono::high_resolution_clock::now();
    int64_t duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - _start).count();
    {
        std::lock_guard<std::mutex> lock(_time_totals_mutex);
        _time_totals[_key] += duration;
    }
}

void ResetPerformanceCounters()
{
    std::lock_guard<std::mutex> lock(_time_totals_mutex);
    _time_totals.clear();
}

std::string TotalPerformanceSummary()
{
    using entry_t = std::pair<int64_t, std::string_view>;
    std::vector<entry_t> entries;

    // copy from 'live' data to a vector
    {
        std::lock_guard<std::mutex> lock(_time_totals_mutex);
        entries.reserve(_time_totals.size());
        for (const auto &e : _time_totals)
        {
            entries.emplace_back(e.second, e.first);
        }
    }

    // sort by time
    std::sort(entries.begin(), entries.end());
    std::reverse(entries.begin(), entries.end());

    // put into output
    std::ostringstream ss;
    ss << "=====================" << std::endl;
    ss << " Performance summary" << std::endl;
    for (const auto &e : entries)
    {
        ss << std::setw(24) << e.second << ": ";
        ss << std::fixed << std::setw(11) << std::setprecision(3) << e.first * 1e-9 << "s" << std::endl;
    }
    ss << "=====================" << std::endl;

    return ss.str();
}

} // namespace opencalibration
