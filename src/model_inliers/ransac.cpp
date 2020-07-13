#include <opencalibration/model_inliers/ransac.hpp>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

namespace opencalibration
{

void homography_model::fit(const std::array<correspondence, MINIMUM_POINTS> &corrs)
{
    Eigen::Matrix<double, 9, 9> P;
    Eigen::Matrix<double, 2, 9> p_i;

    auto x = [&corrs](size_t i) { return corrs[i].measurement1.x() / corrs[i].measurement1.z(); };
    auto y = [&corrs](size_t i) { return corrs[i].measurement1.y() / corrs[i].measurement1.z(); };

    auto x_ = [&corrs](size_t i) { return corrs[i].measurement2.x() / corrs[i].measurement2.z(); };
    auto y_ = [&corrs](size_t i) { return corrs[i].measurement2.y() / corrs[i].measurement2.z(); };

    for (size_t i = 0; i < 4; i++)
    {
        // clang-format off
        p_i << -x(i), -y(i), -1, 0, 0, 0, x(i)*x_(i), y(i)*x_(i), x_(i),
               0, 0, 0, -x(i), -y(i), -1, x(i)*y_(i), y(i)*y_(i), y_(i);
        // clang-format on
        P.block<2, 9>(i * 2, 0) = p_i;
    }
    P.bottomRows<1>() << 0, 0, 0, 0, 0, 0, 0, 0, 1;

    Eigen::Matrix<double, 9, 1> rhs = P.bottomRows<1>();

    P.bottomRows<1>() = rhs.transpose();

    rhs = P.lu().solve(rhs);
    homography.row(0) = rhs.topRows<3>();
    homography.row(1) = rhs.middleRows<3>(3);
    homography.row(2) = rhs.bottomRows<3>();
}

double homography_model::error(const correspondence &corr)
{
    return ((homography * corr.measurement1.hnormalized().homogeneous()).hnormalized() -
            corr.measurement2.hnormalized())
        .norm();
}

template <int n> double fast_pow(double d);

template <> inline double fast_pow<4>(double d)
{
    return (d * d) * (d * d);
}

template <typename Model>
double ransac(const std::vector<correspondence> &matches, Model &model, std::vector<bool> &inliers)
{
    const double INLIER_THRESHOLD = 0.01;
    const size_t MAX_ITERATIONS = 500;
    const double PROBABILITY = 0.999;

    const double log_1m_p = std::log(1 - PROBABILITY);

    if (matches.size() < Model::MINIMUM_POINTS)
    {
        return 0; // need at least this much score increase
    }

    inliers.resize(matches.size());

    Model best_model;
    double best_inlier_count = 0;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, matches.size());

    int probability_iterations = MAX_ITERATIONS;

    for (size_t i = 0; i < probability_iterations; i++)
    {
        std::array<size_t, Model::MINIMUM_POINTS> initial_indices;

        for (size_t j = 0; j < Model::MINIMUM_POINTS; j++)
        {
            int candidate = distribution(generator);
            bool unique = true;
            for (size_t k = 0; k < j; k++)
            {
                if (initial_indices[k] == candidate)
                {
                    unique = false;
                    break;
                }
            }
            if (unique)
            {
                initial_indices[j] = candidate;
            }
            else
            {
                j--;
            }
        }

        std::array<correspondence, Model::MINIMUM_POINTS> initial_values;
        for (size_t j = 0; j < Model::MINIMUM_POINTS; j++)
        {
            initial_values[j] = matches[initial_indices[j]];
        }
        model.fit(initial_values);
        size_t inlier_count = 0;
        for (size_t j = 0; j < matches.size(); j++)
        {
            double error = model.error(matches[j]);
            inlier_count += error < INLIER_THRESHOLD;
        }

        if (inlier_count > best_inlier_count)
        {
            best_model = model;
            best_inlier_count = inlier_count;

            double omega = (double)inlier_count / matches.size();
            double omega_n = fast_pow<Model::MINIMUM_POINTS>(omega);
            double log_1m_omega_n = std::log(1 - omega_n);
            probability_iterations = std::min(MAX_ITERATIONS, static_cast<size_t>(log_1m_p / log_1m_omega_n));
        }
    }

    model = best_model;
    for (size_t j = 0; j < matches.size(); j++)
    {
        double error = model.error(matches[j]);
        inliers[j] = error < INLIER_THRESHOLD;
    }
    return (double)best_inlier_count / matches.size();
}

template double ransac<homography_model>(const std::vector<correspondence> &, homography_model &, std::vector<bool> &);

} // namespace opencalibration
