#pragma once

#include <chrono>
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

  private:
    std::string_view _key;
    std::chrono::time_point<std::chrono::system_clock> _start;
};

void ResetPerformanceCounters();
std::string TotalPerformanceSummary();

} // namespace opencalibration
