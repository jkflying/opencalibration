#include <opencalibration/model_inliers/ransac.hpp>

#include <random>

#include <iostream>

namespace opencalibration
{

template <int n> double fast_pow(double d);

template <> inline double fast_pow<4>(double d)
{
    double t = d * d;
    return t * t;
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
    const size_t MAX_ITERATIONS = 500;
    const size_t MAX_INNER_ITERATIONS = 5;
    const double PROBABILITY = 0.999;

    const double log_1m_p = std::log(1 - PROBABILITY);

    inliers.resize(matches.size());
    std::fill(inliers.begin(), inliers.end(), false);

    if (matches.size() < Model::MINIMUM_POINTS)
    {
        return 0; // need at least this much score increase
    }

    Model best_model;
    double best_inlier_count = 0;

    std::default_random_engine generator(42);

    auto random_k_from_n = [&generator](int n) {
        std::array<size_t, Model::MINIMUM_POINTS> indices;
        std::uniform_int_distribution<size_t> distribution(0, n - 1);
        for (size_t j = 0; j < Model::MINIMUM_POINTS; j++)
        {
            size_t candidate = distribution(generator);
            bool unique = true;
            for (size_t k = 0; k < j; k++)
            {
                if (indices[k] == candidate)
                {
                    unique = false;
                    break;
                }
            }
            if (unique)
            {
                indices[j] = candidate;
            }
            else
            {
                j--;
            }
        }
        return indices;
    };

    size_t probability_iterations = MAX_ITERATIONS;

    for (size_t i = 0; i < probability_iterations; i++)
    {
        std::array<size_t, Model::MINIMUM_POINTS> initial_indices = random_k_from_n(matches.size());

        model.fit(matches, initial_indices);
        size_t inlier_count = model.evaluate(matches, inliers);

        if (inlier_count > best_inlier_count)
        {
            best_model = model;
            best_inlier_count = inlier_count;

            for (size_t j = 0; j < MAX_INNER_ITERATIONS; j++)
            {
                model.fitInliers(matches, inliers);
                inlier_count = model.evaluate(matches, inliers);
                if (inlier_count > best_inlier_count)
                {
                    best_model = model;
                    best_inlier_count = inlier_count;
                }
                else
                {
                    break;
                }
            }

            double omega = static_cast<double>(best_inlier_count) / matches.size();
            double omega_n = fast_pow<Model::MINIMUM_POINTS>(omega);
            double log_1m_omega_n = std::log(1 - omega_n);
            probability_iterations =
                std::max(MIN_ITERATIONS, std::min(MAX_ITERATIONS, static_cast<size_t>(log_1m_p / log_1m_omega_n)));
        }
    }

    model = best_model;
    return (double)model.evaluate(matches, inliers) / matches.size();
}

template double ransac(const std::vector<correspondence> &, homography_model &, std::vector<bool> &);
template double ransac(const std::vector<correspondence> &, fundamental_matrix_model &, std::vector<bool> &);

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
