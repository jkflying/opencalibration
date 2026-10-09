#include <opencalibration/performance/performance.hpp>

#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <atomic>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;
using NodeKey = std::pair<int, std::string_view>;
using NodeIndex = std::map<NodeKey, int>;
using TimeByNode = ankerl::unordered_dense::map<int, int64_t>;
using TimeByKey = ankerl::unordered_dense::map<std::string_view, int64_t>;

constexpr int ROOT = -1;
constexpr int64_t BIN_NS = 100'000'000;

struct Node
{
    int parent;
    std::string_view key;
};

struct Bin
{
    TimeByNode self_ns;
    int64_t first_ns = std::numeric_limits<int64_t>::max();
    int64_t last_ns = std::numeric_limits<int64_t>::min();

    void add(int node, int64_t self, int64_t start_ns, int64_t end_ns)
    {
        self_ns[node] += self;
        first_ns = std::min(first_ns, start_ns);
        last_ns = std::max(last_ns, end_ns);
    }

    int64_t totalNs() const
    {
        int64_t total = 0;
        for (const auto &[node, ns] : self_ns)
            total += ns;
        return total;
    }

    void merge(const Bin &other)
    {
        for (const auto &[node, ns] : other.self_ns)
            self_ns[node] += ns;
        first_ns = std::min(first_ns, other.first_ns);
        last_ns = std::max(last_ns, other.last_ns);
    }
};

struct ThreadLog
{
    std::mutex mutex;
    int64_t bin_index = -1;
    Bin bin;
};

std::atomic<bool> _enable_counters = false;
std::map<int64_t, Bin> _bins;
std::vector<std::shared_ptr<ThreadLog>> _thread_logs;
std::vector<Node> _nodes;
NodeIndex _node_index;
std::mutex _globals_mutex;

thread_local opencalibration::PerformanceMeasure *_innermost = nullptr;

int64_t nanosecondsSinceEpoch(Clock::time_point t)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

ThreadLog &threadLog()
{
    thread_local std::shared_ptr<ThreadLog> log = [] {
        auto l = std::make_shared<ThreadLog>();
        std::lock_guard<std::mutex> lock(_globals_mutex);
        _thread_logs.push_back(l);
        return l;
    }();
    return *log;
}

int nodeFor(int parent, std::string_view key)
{
    thread_local NodeIndex thread_index;
    const NodeKey node_key{parent, key};
    if (auto it = thread_index.find(node_key); it != thread_index.end())
        return it->second;

    std::lock_guard<std::mutex> lock(_globals_mutex);
    const auto [it, inserted] = _node_index.try_emplace(node_key, static_cast<int>(_nodes.size()));
    if (inserted)
        _nodes.push_back({parent, key});
    thread_index.emplace(node_key, it->second);
    return it->second;
}

void record(int node, int64_t self_ns, int64_t start_ns, int64_t end_ns)
{
    const int64_t start_bin = start_ns / BIN_NS;
    const int64_t end_bin = end_ns / BIN_NS;
    ThreadLog &log = threadLog();
    std::lock_guard<std::mutex> lock(log.mutex);
    if (start_bin == end_bin && start_bin == log.bin_index)
    {
        log.bin.add(node, self_ns, start_ns, end_ns);
        return;
    }

    auto selfShare = [&](int64_t from, int64_t to) {
        return static_cast<int64_t>(static_cast<double>(self_ns) * static_cast<double>(to - from) /
                                    static_cast<double>(end_ns - start_ns));
    };

    std::lock_guard<std::mutex> globals_lock(_globals_mutex);
    if (log.bin_index >= 0)
        _bins[log.bin_index].merge(log.bin);
    log.bin = Bin{};
    log.bin_index = end_bin;
    for (int64_t b = start_bin; b < end_bin; b++)
    {
        const int64_t from = std::max(start_ns, b * BIN_NS);
        _bins[b].add(node, selfShare(from, (b + 1) * BIN_NS), from, (b + 1) * BIN_NS);
    }
    const int64_t from = std::max(start_ns, end_bin * BIN_NS);
    log.bin.add(node, selfShare(from, end_ns), from, end_ns);
}

std::map<int64_t, Bin> snapshotBins()
{
    std::map<int64_t, Bin> bins;
    std::vector<std::shared_ptr<ThreadLog>> logs;
    {
        std::lock_guard<std::mutex> lock(_globals_mutex);
        bins = _bins;
        logs = _thread_logs;
    }
    for (const auto &log : logs)
    {
        std::lock_guard<std::mutex> lock(log->mutex);
        if (log->bin_index >= 0)
            bins[log->bin_index].merge(log->bin);
    }
    return bins;
}

std::vector<Node> snapshotNodes()
{
    std::lock_guard<std::mutex> lock(_globals_mutex);
    return _nodes;
}

TimeByNode selfTimeByNode(const std::map<int64_t, Bin> &bins)
{
    TimeByNode totals;
    for (const auto &[index, bin] : bins)
        for (const auto &[node, ns] : bin.self_ns)
            totals[node] += ns;
    return totals;
}

TimeByNode selfWallTimeByNode(const std::map<int64_t, Bin> &bins)
{
    TimeByNode wall_time;
    for (const auto &[index, bin] : bins)
    {
        const int64_t bin_total = bin.totalNs();
        if (bin_total <= 0)
            continue;
        const int64_t busy = std::min(bin_total, bin.last_ns - bin.first_ns);
        for (const auto &[node, ns] : bin.self_ns)
            wall_time[node] += static_cast<int64_t>(static_cast<double>(busy) * static_cast<double>(ns) /
                                                    static_cast<double>(bin_total));
    }
    return wall_time;
}

TimeByKey selfTimeByKey(const std::vector<Node> &nodes, const TimeByNode &self_by_node)
{
    TimeByKey totals;
    for (const auto &[node, ns] : self_by_node)
        totals[nodes[node].key] += ns;
    return totals;
}

struct SummaryRow
{
    size_t depth;
    std::string_view key;
    int64_t system_ns, wall_ns, self_ns;
};

struct SummaryTree
{
    const std::vector<Node> &nodes;
    const TimeByNode &self;
    const TimeByNode &self_wall;
    std::vector<std::vector<int>> children;
    std::vector<int64_t> inclusive_system, inclusive_wall;

    SummaryTree(const std::vector<Node> &nodes_, const TimeByNode &self_, const TimeByNode &self_wall_)
        : nodes(nodes_), self(self_), self_wall(self_wall_), children(nodes_.size()),
          inclusive_system(nodes_.size(), 0), inclusive_wall(nodes_.size(), 0)
    {
        for (size_t i = nodes.size(); i-- > 0;)
        {
            inclusive_system[i] += valueOr(self, i);
            inclusive_wall[i] += valueOr(self_wall, i);
            if (nodes[i].parent != ROOT)
            {
                inclusive_system[nodes[i].parent] += inclusive_system[i];
                inclusive_wall[nodes[i].parent] += inclusive_wall[i];
            }
        }
        for (size_t i = 0; i < nodes.size(); i++)
            if (nodes[i].parent != ROOT)
                children[nodes[i].parent].push_back(static_cast<int>(i));
    }

    std::vector<SummaryRow> rows() const
    {
        std::vector<int> roots;
        for (size_t i = 0; i < nodes.size(); i++)
            if (nodes[i].parent == ROOT)
                roots.push_back(static_cast<int>(i));

        std::vector<SummaryRow> out;
        appendSorted(roots, 0, out);
        return out;
    }

  private:
    static int64_t valueOr(const TimeByNode &m, size_t node)
    {
        const auto it = m.find(static_cast<int>(node));
        return it == m.end() ? 0 : it->second;
    }

    void appendSorted(std::vector<int> siblings, size_t depth, std::vector<SummaryRow> &out) const
    {
        std::sort(siblings.begin(), siblings.end(),
                  [&](int a, int b) { return inclusive_system[a] > inclusive_system[b]; });
        for (int node : siblings)
        {
            if (inclusive_system[node] <= 0)
                continue;
            out.push_back({depth, nodes[node].key, inclusive_system[node], inclusive_wall[node],
                           valueOr(self, static_cast<size_t>(node))});
            appendSorted(children[node], depth + 1, out);
        }
    }
};

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
    if (key.len > 0)
        initialize(key);
}

void PerformanceMeasure::initialize(const opencalibration::Literal &key)
{
    if (!_enable_counters.load(std::memory_order_relaxed))
    {
        return;
    }

    _parent = _innermost;
    _node = nodeFor(_parent ? _parent->_node : ROOT, std::string_view(key.ptr, key.len));
    _child_ns = 0;
    _innermost = this;
    _start = Clock::now();
    _running = true;
}

void PerformanceMeasure::finalize()
{
    const bool was_running = std::exchange(_running, false);
    if (!was_running)
    {
        return;
    }

    const int64_t start_ns = nanosecondsSinceEpoch(_start);
    const int64_t end_ns = nanosecondsSinceEpoch(Clock::now());
    _innermost = _parent;
    if (_parent)
        _parent->_child_ns += end_ns - start_ns;
    record(_node, std::max<int64_t>(0, end_ns - start_ns - _child_ns), start_ns, end_ns);
}

std::string TopPerformanceTotalsSinceLastCall(size_t max_entries)
{
    static TimeByKey previous;
    const auto totals = selfTimeByKey(snapshotNodes(), selfTimeByNode(snapshotBins()));

    std::vector<std::pair<int64_t, std::string_view>> entries;
    for (const auto &[key, total] : totals)
    {
        const int64_t delta = total - std::exchange(previous[key], total);
        if (delta > 0)
            entries.emplace_back(delta, key);
    }
    std::sort(entries.rbegin(), entries.rend());
    entries.resize(std::min(entries.size(), max_entries));

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    for (const auto &[nanoseconds, key] : entries)
        ss << (ss.tellp() > 0 ? ", " : "") << key << " " << nanoseconds * 1e-9 << "s";
    return ss.str();
}

std::string TotalPerformanceSummary()
{
    const auto bins = snapshotBins();
    const auto nodes = snapshotNodes();
    const auto self = selfTimeByNode(bins);
    const auto self_wall = selfWallTimeByNode(bins);
    const auto rows = SummaryTree(nodes, self, self_wall).rows();

    std::ostringstream ss;
    ss << "=====================" << std::endl;
    ss << " Performance summary" << std::endl;

    ss << std::setw(25) << "Key" << std::setw(15) << "System" << std::setw(15) << "Wall" << std::setw(14)
       << "Parallelism" << std::setw(14) << "Self" << std::endl;
    for (const auto &row : rows)
    {
        ss << std::setw(24) << (std::string(2 * row.depth, ' ') + std::string(row.key)) << ":";
        ss << std::fixed << std::setw(14) << std::setprecision(3) << row.system_ns * 1e-9 << "s";
        ss << std::fixed << std::setw(14) << std::setprecision(3) << row.wall_ns * 1e-9 << "s";
        ss << std::fixed << std::setw(14) << std::setprecision(3) << row.system_ns / (double)row.wall_ns;
        ss << std::fixed << std::setw(13) << std::setprecision(3) << row.self_ns * 1e-9 << "s";
        ss << std::endl;
    }
    ss << "=====================" << std::endl;

    return ss.str();
}

} // namespace opencalibration
