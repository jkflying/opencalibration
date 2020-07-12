#include <opencalibration/match/match_features.hpp>

namespace
{
double descriptor_distance(const opencalibration::feature_2d &f1, const opencalibration::feature_2d &f2)
{
    return (f1.descriptor ^ f2.descriptor).count() * (1.0 / opencalibration::feature_2d::DESCRIPTOR_BITS);
}

} // namespace

namespace opencalibration
{
std::vector<feature_match> match_features(const std::vector<feature_2d> &set_1, const std::vector<feature_2d> &set_2)
{
    std::vector<feature_match> results;
    results.reserve(set_1.size());

    for (size_t i = 0; i < set_1.size(); i++)
    {
        const auto &f1 = set_1[i];
        feature_match best_match{i, 0, std::numeric_limits<double>::infinity()};
        double second_best_distance = std::numeric_limits<double>::infinity();

        for (size_t j = 0; j < set_2.size(); j++)
        {
            const auto &f2 = set_2[j];
            double distance = descriptor_distance(f1, f2);
            if (distance < second_best_distance)
            {
                if (distance < best_match.distance)
                {
                    second_best_distance = best_match.distance;
                    best_match.distance = distance;
                    best_match.feature_index_2 = j;
                }
                else
                {
                    second_best_distance = distance;
                }
            }
        }
        if (best_match.distance * 1.5 < second_best_distance)
        {
            results.push_back(best_match);
        }
    }

    return results;
}
} // namespace opencalibration
