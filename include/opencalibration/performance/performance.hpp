#pragma once

#include <chrono>
#include <cstdint>
#include <string>

namespace opencalibration
{

struct Literal
{
    template <std::size_t N> Literal(const char (&literal)[N] = "") : ptr(literal), len(N - 1)
    {
    }

    const char *const ptr;
    const size_t len;
};

class PerformanceMeasure
{
  public:
    PerformanceMeasure(const Literal &key);
    ~PerformanceMeasure();
    void reset(const Literal &key);

  private:
    void initialize(const Literal &key);
    void finalize();
    bool _running = false;
    int _node = -1;
    PerformanceMeasure *_parent = nullptr;
    int64_t _child_ns = 0;
    std::chrono::time_point<std::chrono::steady_clock> _start;
};

void EnablePerformanceCounters(bool enable);
std::string TotalPerformanceSummary();
std::string TopPerformanceTotalsSinceLastCall(size_t max_entries);

} // namespace opencalibration
