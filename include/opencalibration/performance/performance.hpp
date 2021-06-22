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
    void reset(const Literal &key);

  private:
    void initialize(const Literal &key);
    void finalize();
    bool _running;
    std::string_view _key;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
};

void EnablePerformanceCounters(bool enable);
void ResetPerformanceCounters();
std::string TotalPerformanceSummary();

} // namespace opencalibration
