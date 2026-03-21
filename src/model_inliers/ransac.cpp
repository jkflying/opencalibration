#include <opencalibration/model_inliers/ransac.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <type_traits>

namespace opencalibration
{

template <typename T, typename = void> struct has_check_sample_degeneracy : std::false_type
{
};
template <typename T>
struct has_check_sample_degeneracy<
    T, std::void_t<decltype(T::checkSampleDegeneracy(std::declval<const std::vector<correspondence> &>(),
                                                     std::declval<const std::array<size_t, T::MINIMUM_POINTS> &>()))>>
    : std::true_type
{
};

template <typename T, typename = void> struct has_check_degeneracy : std::false_type
{
};
template <typename T>
struct has_check_degeneracy<
    T, std::void_t<decltype(std::declval<T>().checkDegeneracy(std::declval<const std::vector<correspondence> &>(),
                                                              std::declval<std::vector<bool> &>()))>> : std::true_type
{
};

template <int n> double fast_pow(double d);

template <> inline double fast_pow<4>(double d)
{
    double t = d * d;
    return t * t;
}

template <> inline double fast_pow<5>(double d)
{
    double t = d * d;
    return t * t * d;
}

template <> inline double fast_pow<8>(double d)
{
    double t = d * d;
    t = t * t;
    return t * t;
}

template <typename Model>
double ransac(const std::vector<correspondence> &matches, Model &model, std::vector<bool> &inliers)
{

    const size_t MIN_ITERATIONS = 20;
    const size_t MAX_ITERATIONS = 10000;
    const size_t MAX_INNER_ITERATIONS = 5;
    const double PROBABILITY = 0.999;

    const double log_1m_p = std::log(1 - PROBABILITY);

    inliers.resize(matches.size());
    std::fill(inliers.begin(), inliers.end(), false);

    if (matches.size() < Model::MINIMUM_POINTS)
    {
        return 0; // need at least this much score increase
    }

    bool has_quality = false;
    for (const auto &m : matches)
    {
        if (m.quality != 0)
        {
            has_quality = true;
            break;
        }
    }

    // PROSAC: sort by match quality so early iterations sample better correspondences
    std::vector<size_t> sorted_idx;
    if (has_quality)
    {
        sorted_idx.resize(matches.size());
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&matches](size_t a, size_t b) { return matches[a].quality < matches[b].quality; });
    }

    std::vector<size_t> eval_order(matches.size());
    std::iota(eval_order.begin(), eval_order.end(), 0);

    Model best_model{};
    double best_score = 0;

    std::default_random_engine generator(42);

    size_t prosac_n = has_quality ? Model::MINIMUM_POINTS : matches.size();

    auto map_idx = [&sorted_idx, has_quality](size_t i) -> size_t { return has_quality ? sorted_idx[i] : i; };

    auto random_k_from_n = [&generator, &map_idx](size_t pool) {
        std::array<size_t, Model::MINIMUM_POINTS> indices;
        std::uniform_int_distribution<size_t> dist(0, pool - 1);
        for (size_t j = 0; j < Model::MINIMUM_POINTS; j++)
        {
            size_t candidate;
            bool unique;
            do
            {
                candidate = dist(generator);
                unique = true;
                for (size_t k = 0; k < j; k++)
                {
                    if (indices[k] == map_idx(candidate))
                    {
                        unique = false;
                        break;
                    }
                }
            } while (!unique);
            indices[j] = map_idx(candidate);
        }
        return indices;
    };

    // PROSAC growth-point sampling: always includes the newest point in the pool
    auto prosac_sample = [&generator, &sorted_idx](size_t pool) {
        std::array<size_t, Model::MINIMUM_POINTS> indices;
        indices[0] = sorted_idx[pool - 1];
        std::uniform_int_distribution<size_t> dist(0, pool - 2);
        for (size_t j = 1; j < Model::MINIMUM_POINTS; j++)
        {
            size_t candidate;
            bool unique;
            do
            {
                candidate = dist(generator);
                unique = true;
                for (size_t k = 0; k < j; k++)
                {
                    if (indices[k] == sorted_idx[candidate])
                    {
                        unique = false;
                        break;
                    }
                }
            } while (!unique);
            indices[j] = sorted_idx[candidate];
        }
        return indices;
    };

    size_t probability_iterations = MAX_ITERATIONS;

    std::shuffle(eval_order.begin(), eval_order.end(), generator);

    std::vector<bool> candidate_inliers(matches.size(), false);

    for (size_t i = 0; i < probability_iterations; i++)
    {
        if (has_quality && prosac_n < matches.size() && i > 0 && i % 10 == 0)
            prosac_n++;

        std::array<size_t, Model::MINIMUM_POINTS> initial_indices;
        if (has_quality && prosac_n < matches.size() && prosac_n > Model::MINIMUM_POINTS)
            initial_indices = prosac_sample(prosac_n);
        else
            initial_indices = random_k_from_n(has_quality ? prosac_n : matches.size());

        if constexpr (has_check_sample_degeneracy<Model>::value)
        {
            if (Model::checkSampleDegeneracy(matches, initial_indices))
                continue;
        }

        model.fit(matches, initial_indices);

        // SPRT: evaluate in shuffled order, reject early if clearly worse than best
        // MSAC scoring (1 - (e/t)^2) must match Model::evaluate()
        double score = 0;
        size_t checked = 0;
        bool rejected = false;
        std::fill(candidate_inliers.begin(), candidate_inliers.end(), false);
        for (size_t idx : eval_order)
        {
            double e = model.error(matches[idx]);
            if (e < model.inlier_threshold)
            {
                candidate_inliers[idx] = true;
                double ratio = e / model.inlier_threshold;
                score += 1.0 - ratio * ratio;
            }
            checked++;
            if (checked > 20 && best_score > 0 &&
                score < best_score * static_cast<double>(checked) / matches.size() * 0.6)
            {
                rejected = true;
                break;
            }
        }
        if (rejected)
            continue;

        if (score > best_score)
        {
            best_model = model;
            best_score = score;
            inliers = candidate_inliers;

            if constexpr (has_check_degeneracy<Model>::value)
            {
                model.checkDegeneracy(matches, inliers);
                double degen_score = model.evaluate(matches, inliers);
                if (degen_score > best_score)
                {
                    best_model = model;
                    best_score = degen_score;
                }
            }

            model.fitInliers(matches, inliers);
            double inlier_score = model.evaluate(matches, inliers);
            if (inlier_score > best_score)
            {
                best_model = model;
                best_score = inlier_score;

                for (size_t j = 1; j < MAX_INNER_ITERATIONS; j++)
                {
                    model.fitInliers(matches, inliers);
                    inlier_score = model.evaluate(matches, inliers);
                    if (inlier_score > best_score)
                    {
                        best_model = model;
                        best_score = inlier_score;
                    }
                    else
                    {
                        break;
                    }
                }
            }

            double omega = best_score / matches.size();
            double omega_n = fast_pow<Model::MINIMUM_POINTS>(omega);
            double log_1m_omega_n = std::log(1 - omega_n);
            probability_iterations =
                std::max(MIN_ITERATIONS, std::min(MAX_ITERATIONS, static_cast<size_t>(log_1m_p / log_1m_omega_n)));
        }
    }

    model = best_model;
    return model.evaluate(matches, inliers) / matches.size();
}

template double ransac(const std::vector<correspondence> &, homography_model &, std::vector<bool> &);
template double ransac(const std::vector<correspondence> &, fundamental_matrix_model &, std::vector<bool> &);
template double ransac(const std::vector<correspondence> &, essential_matrix_model &, std::vector<bool> &);

void assembleInliers(const std::vector<feature_match> &matches, const std::vector<bool> &inliers,
                     const std::vector<feature_2d> &source_features, const std::vector<feature_2d> &dest_features,
                     std::vector<feature_match_denormalized> &inlier_list)
{

    inlier_list.reserve(std::count(inliers.begin(), inliers.end(), true));
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (inliers[i])
        {
            feature_match_denormalized fmd;
            fmd.pixel_1 = source_features[matches[i].feature_index_1].location;
            fmd.pixel_2 = dest_features[matches[i].feature_index_2].location;
            fmd.feature_index_1 = matches[i].feature_index_1;
            fmd.feature_index_2 = matches[i].feature_index_2;
            fmd.match_index = i;
            inlier_list.push_back(fmd);
        }
    }
}

} // namespace opencalibration
