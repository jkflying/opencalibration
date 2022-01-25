#include <opencalibration/io/saveXYZ.hpp>

namespace opencalibration
{

bool toXYZ(const std::vector<surface_model> &surfaces, std::ostream &out,
           const std::array<std::pair<int64_t, int64_t>, 3> &bounds)
{

    std::function<bool(const Eigen::Vector3d &)> inbounds;

    if (bounds[0].first == bounds[0].second && bounds[1].first == bounds[1].second &&
        bounds[2].first == bounds[2].second)
    {
        inbounds = [](const Eigen::Vector3d &) -> bool { return true; };
    }
    else
    {
        inbounds = [&bounds](const Eigen::Vector3d &v) -> bool {
            bool res = true;
            for (size_t i = 0; i < bounds.size(); i++)
            {
                res &= bounds[i].first < v[i] && v[i] < bounds[i].second;
            }
            return res;
        };
    }

    std::ostringstream buffer;

    for (const auto &s : surfaces)
    {
        for (const auto &c : s.cloud)
        {
            for (const auto &p : c)
            {
                if (inbounds(p))
                {
                    buffer << p.x() << "," << p.y() << "," << p.z() << "\n";
                }
            }
            out << buffer.str();
            buffer.str("");
            buffer.clear();
        }
    }
    return true;
}

std::array<std::pair<int64_t, int64_t>, 3> filterOutliers(const std::vector<surface_model> &surfaces)
{
    std::array<std::unordered_map<int64_t, size_t>, 3> count_map{};

    size_t total = 0;
    for (const auto &s : surfaces)
    {
        for (const auto &c : s.cloud)
        {
            for (const auto &p : c)
            {
                for (size_t i = 0; i < count_map.size(); i++)
                {
                    count_map[i][static_cast<int64_t>(p[i])]++;
                }
                total++;
            }
        }
    }

    auto dimbox = [](const std::unordered_map<int64_t, size_t> &count_map,
                     size_t total) -> std::pair<int64_t, int64_t> {
        std::vector<std::pair<int64_t, size_t>> counts;
        counts.insert(counts.end(), count_map.begin(), count_map.end());
        std::sort(counts.begin(), counts.end());

        const size_t cutoff = total * 0.025;

        size_t lowSum = 0, lowIndex = 0;
        while (lowIndex < counts.size() && lowSum < cutoff)
        {
            lowSum += counts[lowIndex++].second;
        }

        size_t highSum = 0, highIndex = counts.size() - 1;
        while (highIndex > lowIndex && highSum < cutoff)
        {
            highSum += counts[highIndex--].second;
        }

        int64_t lowBound = counts.empty() ? 0 : counts[lowIndex].first,
                highBound = counts.empty() ? 0 : counts[highIndex].first;
        int64_t width = (highBound - lowBound) * 2;
        int64_t mid = lowBound + width / 2;

        return {mid - width, mid + width};
    };

    return {dimbox(count_map[0], total), dimbox(count_map[1], total), dimbox(count_map[2], total)};
}

} // namespace opencalibration
