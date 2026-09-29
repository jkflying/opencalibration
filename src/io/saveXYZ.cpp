#include <opencalibration/io/saveXYZ.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace opencalibration
{

bool toXYZ(const std::vector<surface_model> &surfaces, std::ostream &out,
           const std::array<std::pair<int64_t, int64_t>, 3> &bounds)
{
    const bool unbounded = std::all_of(bounds.begin(), bounds.end(), [](const auto &b) { return b.first == b.second; });
    auto inbounds = [&bounds, unbounded](const Eigen::Vector3d &v) {
        for (size_t i = 0; i < bounds.size(); i++)
            if (!unbounded && !(bounds[i].first <= v[i] && v[i] < bounds[i].second))
                return false;
        return true;
    };

    out.precision(std::numeric_limits<double>::max_digits10);
    for (const auto &s : surfaces)
        for (const auto &c : s.cloud)
            for (const auto &p : c)
                if (inbounds(p))
                    out << p.x() << "," << p.y() << "," << p.z() << "\n";
    return true;
}

std::array<std::pair<int64_t, int64_t>, 3> filterOutliers(const std::vector<surface_model> &surfaces)
{
    std::array<ankerl::unordered_dense::map<int64_t, size_t>, 3> count_map{};

    size_t total = 0;
    for (const auto &s : surfaces)
    {
        for (const auto &c : s.cloud)
        {
            for (const auto &p : c)
            {
                for (size_t i = 0; i < count_map.size(); i++)
                {
                    count_map[i][static_cast<int64_t>(std::floor(p[i]))]++;
                }
                total++;
            }
        }
    }

    auto dimbox = [](const ankerl::unordered_dense::map<int64_t, size_t> &count_map,
                     size_t total) -> std::pair<int64_t, int64_t> {
        std::vector<std::pair<int64_t, size_t>> counts;
        counts.insert(counts.end(), count_map.begin(), count_map.end());
        std::sort(counts.begin(), counts.end());

        if (counts.empty())
            return {0, 0};

        const size_t cutoff = total * 0.025;

        size_t lowSum = 0, lowIndex = 0;
        while (lowIndex < counts.size() && lowSum < cutoff)
        {
            lowSum += counts[lowIndex++].second;
        }

        if (lowIndex > 0)
            lowIndex--;

        size_t highSum = 0, highIndex = counts.size() - 1;
        while (highIndex > lowIndex && highSum < cutoff)
        {
            highSum += counts[highIndex--].second;
        }

        const int64_t lowBound = counts[lowIndex].first;
        const int64_t highBound = counts[highIndex].first + 1;
        const int64_t extent = highBound - lowBound;
        return {lowBound - extent, highBound + extent};
    };

    return {dimbox(count_map[0], total), dimbox(count_map[1], total), dimbox(count_map[2], total)};
}

} // namespace opencalibration
