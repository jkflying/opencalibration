#include <opencalibration/match/match_features.hpp>

#include <jk/KDTree.h>

#include <eigen3/Eigen/Geometry>

#include <unordered_set>

namespace
{
double descriptor_distance(const opencalibration::feature_2d &f1, const opencalibration::feature_2d &f2)
{
    return (f1.descriptor ^ f2.descriptor).count() * (1.0 / opencalibration::feature_2d::DESCRIPTOR_BITS);
}

} // namespace

namespace opencalibration
{

std::vector<size_t> spatially_subsample_feature_indices(const std::vector<feature_2d> &features, double spacing_pixels)
{
    if (features.empty())
        return {};

    std::vector<size_t> sorted_indices(features.size());
    for (size_t i = 0; i < features.size(); i++)
    {
        sorted_indices[i] = i;
    }
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&features](size_t a, size_t b) { return features[a].strength > features[b].strength; });

    std::vector<size_t> indices;
    indices.reserve(features.size() / 4);

    auto toArray = [](const Eigen::Vector2d &v) -> std::array<double, 2> { return {v.x(), v.y()}; };
    jk::tree::KDTree<size_t, 2, 8> tree;

    for (size_t idx : sorted_indices)
    {
        const auto &f = features[idx];
        if (tree.size() == 0)
        {
            tree.addPoint(toArray(f.location), indices.size());
            indices.push_back(idx);
        }
        else
        {
            auto searcher = tree.searcher();
            const auto &nn = searcher.search(toArray(f.location), std::numeric_limits<double>::infinity(), 1);
            if (nn[0].distance > spacing_pixels * spacing_pixels)
            {
                tree.addPoint(toArray(f.location), indices.size());
                indices.push_back(idx);
            }
        }
    }

    return indices;
}

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
        if (best_match.distance < 0.8 * second_best_distance)
        {
            results.push_back(best_match);
        }
    }

    std::sort(results.begin(), results.end(),
              [](const feature_match &f1, const feature_match &f2) -> bool { return f1.distance > f2.distance; });
    return results;
}

std::vector<feature_match> match_features_subset(const std::vector<feature_2d> &set_1,
                                                 const std::vector<feature_2d> &set_2,
                                                 const std::vector<size_t> &indices_1,
                                                 const std::vector<size_t> &indices_2)
{
    using descriptor_t = std::bitset<feature_2d::DESCRIPTOR_BITS>;

    // Pack subset descriptors contiguously for cache-friendly inner loop
    std::vector<descriptor_t> packed_2(indices_2.size());
    for (size_t k = 0; k < indices_2.size(); k++)
    {
        packed_2[k] = set_2[indices_2[k]].descriptor;
    }

    std::vector<feature_match> results;
    results.reserve(indices_1.size());

    for (size_t i : indices_1)
    {
        const descriptor_t &desc1 = set_1[i].descriptor;
        feature_match best_match{i, 0, std::numeric_limits<double>::infinity()};
        double second_best_distance = std::numeric_limits<double>::infinity();

        for (size_t k = 0; k < packed_2.size(); k++)
        {
            double distance = (desc1 ^ packed_2[k]).count() * (1.0 / feature_2d::DESCRIPTOR_BITS);
            if (distance < second_best_distance)
            {
                if (distance < best_match.distance)
                {
                    second_best_distance = best_match.distance;
                    best_match.distance = distance;
                    best_match.feature_index_2 = indices_2[k];
                }
                else
                {
                    second_best_distance = distance;
                }
            }
        }
        if (best_match.distance < 0.8 * second_best_distance)
        {
            results.push_back(best_match);
        }
    }

    std::sort(results.begin(), results.end(),
              [](const feature_match &f1, const feature_match &f2) -> bool { return f1.distance > f2.distance; });
    return results;
}

std::vector<feature_match> match_features_local_guided(const std::vector<feature_2d> &set_1,
                                                       const std::vector<feature_2d> &set_2,
                                                       const Eigen::Matrix3d &homography, double search_radius_pixels,
                                                       const Eigen::Matrix3d *fundamental_matrix,
                                                       double epipolar_threshold_pixels)
{
    const double ratio_threshold = 0.8;

    // Epipolar line distance: |x2' * F * x1| / ||(Fx1)[0:1]||
    auto epipolar_distance = [](const Eigen::Matrix3d &F, const Eigen::Vector2d &p1, const Eigen::Vector2d &p2) {
        Eigen::Vector3d epiline = F * p1.homogeneous();
        double numerator = std::abs(p2.homogeneous().dot(epiline));
        double denominator = epiline.head<2>().norm();
        return denominator > 0 ? numerator / denominator : std::numeric_limits<double>::infinity();
    };

    auto toArray = [](const Eigen::Vector2d &v) -> std::array<double, 2> { return {v.x(), v.y()}; };
    jk::tree::KDTree<size_t, 2, 8> tree;
    for (size_t i = 0; i < set_2.size(); i++)
    {
        tree.addPoint(toArray(set_2[i].location), i);
    }

    std::vector<feature_match> forward_matches;
    forward_matches.reserve(set_1.size());

    auto searcher = tree.searcher();
    for (size_t i = 0; i < set_1.size(); i++)
    {
        const auto &f1 = set_1[i];

        Eigen::Vector3d predicted = homography * f1.location.homogeneous();
        Eigen::Vector2d predicted_2d = predicted.hnormalized();

        const auto &candidates = searcher.search(toArray(predicted_2d), search_radius_pixels * search_radius_pixels,
                                                 std::numeric_limits<size_t>::max());

        if (candidates.empty())
            continue;

        feature_match best_match{i, 0, std::numeric_limits<double>::infinity()};
        double second_best_distance = std::numeric_limits<double>::infinity();

        for (const auto &candidate : candidates)
        {
            if (fundamental_matrix != nullptr &&
                epipolar_distance(*fundamental_matrix, f1.location, set_2[candidate.payload].location) >
                    epipolar_threshold_pixels)
            {
                continue;
            }

            const auto &f2 = set_2[candidate.payload];
            double distance = descriptor_distance(f1, f2);
            if (distance < second_best_distance)
            {
                if (distance < best_match.distance)
                {
                    second_best_distance = best_match.distance;
                    best_match.distance = distance;
                    best_match.feature_index_2 = candidate.payload;
                }
                else
                {
                    second_best_distance = distance;
                }
            }
        }

        if (best_match.distance < ratio_threshold * second_best_distance)
        {
            forward_matches.push_back(best_match);
        }
    }

    std::unordered_set<size_t> matched_in_set2;
    for (const auto &fm : forward_matches)
    {
        matched_in_set2.insert(fm.feature_index_2);
    }

    jk::tree::KDTree<size_t, 2, 8> tree1;
    for (size_t i = 0; i < set_1.size(); i++)
    {
        tree1.addPoint(toArray(set_1[i].location), i);
    }

    Eigen::Matrix3d inv_homography = homography.inverse();
    Eigen::Matrix3d F_transpose;
    if (fundamental_matrix != nullptr)
    {
        F_transpose = fundamental_matrix->transpose();
    }
    auto searcher1 = tree1.searcher();
    std::vector<size_t> backward_best(set_2.size(), SIZE_MAX);
    for (size_t j : matched_in_set2)
    {
        const auto &f2 = set_2[j];

        Eigen::Vector3d predicted = inv_homography * f2.location.homogeneous();
        Eigen::Vector2d predicted_2d = predicted.hnormalized();

        const auto &candidates = searcher1.search(toArray(predicted_2d), search_radius_pixels * search_radius_pixels,
                                                  std::numeric_limits<size_t>::max());

        double best_distance = std::numeric_limits<double>::infinity();
        for (const auto &candidate : candidates)
        {
            if (fundamental_matrix != nullptr &&
                epipolar_distance(F_transpose, f2.location, set_1[candidate.payload].location) >
                    epipolar_threshold_pixels)
            {
                continue;
            }

            double distance = descriptor_distance(f2, set_1[candidate.payload]);
            if (distance < best_distance)
            {
                best_distance = distance;
                backward_best[j] = candidate.payload;
            }
        }
    }

    std::vector<feature_match> results;
    results.reserve(forward_matches.size());
    for (const auto &fm : forward_matches)
    {
        if (backward_best[fm.feature_index_2] == fm.feature_index_1)
        {
            results.push_back(fm);
        }
    }

    std::sort(results.begin(), results.end(),
              [](const feature_match &f1, const feature_match &f2) -> bool { return f1.distance > f2.distance; });
    return results;
}

} // namespace opencalibration
