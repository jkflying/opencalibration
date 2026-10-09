#include <opencalibration/performance/performance.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <sstream>
#include <thread>
#include <vector>

using namespace opencalibration;

TEST(performance, many_short_scopes_across_threads_report_bounded_wall_time)
{
    // GIVEN: counters enabled and several threads each opening many back-to-back short scopes
    EnablePerformanceCounters(true);
    constexpr int THREADS = 4;
    std::vector<std::thread> threads;
    for (int t = 0; t < THREADS; t++)
        threads.emplace_back([] {
            for (int i = 0; i < 100000; i++)
            {
                PerformanceMeasure p("perf test short scope");
                p.reset("perf test other scope");
            }
        });
    for (auto &t : threads)
        t.join();

    // WHEN: the summary is generated
    const std::string summary = TotalPerformanceSummary();
    EnablePerformanceCounters(false);

    // THEN: the key is reported with a wall time share no greater than its summed thread time
    std::istringstream lines(summary);
    std::string line;
    bool found = false;
    while (std::getline(lines, line))
    {
        if (line.find("perf test short scope:") == std::string::npos)
            continue;
        found = true;
        std::istringstream fields(line.substr(line.find(':') + 1));
        double system = 0, wall = 0;
        char unit = 0;
        fields >> system >> unit >> wall;
        EXPECT_GT(system, 0);
        EXPECT_GT(wall, 0);
        EXPECT_LE(wall, system * 1.0001);
    }
    EXPECT_TRUE(found) << summary;
}

namespace
{
struct SummaryRow
{
    double system = -1, self = -1;
};

SummaryRow rowOf(const std::string &summary, const std::string &key)
{
    std::istringstream lines(summary);
    std::string line;
    while (std::getline(lines, line))
    {
        const auto colon = line.find(key + ":");
        if (colon == std::string::npos)
            continue;
        std::istringstream fields(line.substr(colon + key.size() + 1));
        SummaryRow row;
        double wall, parallelism;
        char unit;
        fields >> row.system >> unit >> wall >> unit >> parallelism >> row.self;
        return row;
    }
    return {};
}
} // namespace

TEST(performance, nested_scope_time_is_counted_once_and_indented_under_its_parent)
{
    // GIVEN: counters enabled and a parent scope that wraps a child scope that sleeps
    EnablePerformanceCounters(true);
    {
        PerformanceMeasure parent("perf test parent");
        PerformanceMeasure child("perf test child");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // WHEN: the summary is generated
    const std::string summary = TotalPerformanceSummary();
    EnablePerformanceCounters(false);

    // THEN: the parent's total is the child's total plus only its own time, and the child is listed indented below it
    const SummaryRow parent = rowOf(summary, "perf test parent");
    const SummaryRow child = rowOf(summary, "perf test child");
    EXPECT_GE(child.system, 0.002);
    EXPECT_NEAR(parent.system, child.system + parent.self, 0.0015);
    EXPECT_NEAR(child.self, child.system, 0.0015);
    EXPECT_LT(summary.find("perf test parent"), summary.find("  perf test child")) << summary;
}
