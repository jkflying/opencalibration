#include <opencalibration/model_inliers/ransac.hpp>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <random>

namespace opencalibration
{

void homography_model::fit(const std::vector<correspondence> &corrs,
                           const std::array<size_t, MINIMUM_POINTS> &initial_indices)
{
    Eigen::Matrix<double, 9, 9> P;

    auto x = [&initial_indices, &corrs](size_t i) {
        return corrs[initial_indices[i]].measurement1.x() / corrs[i].measurement1.z();
    };
    auto y = [&initial_indices, &corrs](size_t i) {
        return corrs[initial_indices[i]].measurement1.y() / corrs[i].measurement1.z();
    };

    auto x_ = [&initial_indices, &corrs](size_t i) {
        return corrs[initial_indices[i]].measurement2.x() / corrs[i].measurement2.z();
    };
    auto y_ = [&initial_indices, &corrs](size_t i) {
        return corrs[initial_indices[i]].measurement2.y() / corrs[i].measurement2.z();
    };

    for (size_t i = 0; i < 4; i++)
    {
        P.row(i * 2) << -x(i), -y(i), -1, 0, 0, 0, x(i) * x_(i), y(i) * x_(i), x_(i);
        P.row(i * 2 + 1) << 0, 0, 0, -x(i), -y(i), -1, x(i) * y_(i), y(i) * y_(i), y_(i);
    }

    // add constraint that bottom right corner is 1
    P.bottomRows<1>().setZero();
    P.bottomRightCorner<1, 1>() << 1;
    Eigen::Matrix<double, 9, 1> rhs;
    rhs.setZero();
    rhs.bottomRows<1>() << 1;

    Eigen::Matrix<double, 9, 1> H_ = P.fullPivLu().solve(rhs);
    homography.row(0) = H_.topRows<3>().transpose();
    homography.row(1) = H_.middleRows<3>(3).transpose();
    homography.row(2) = H_.bottomRows<3>().transpose();
}

void homography_model::fitInliers(const std::vector<correspondence> &corrs, const std::vector<bool> &inliers)
{
    size_t num_inliers = std::count(inliers.begin(), inliers.end(), true);
    Eigen::Matrix<double, Eigen::Dynamic, 9> P(num_inliers * 2 + 1, 9);

    auto x = [&corrs](size_t i) { return corrs[i].measurement1.x() / corrs[i].measurement1.z(); };
    auto y = [&corrs](size_t i) { return corrs[i].measurement1.y() / corrs[i].measurement1.z(); };

    auto x_ = [&corrs](size_t i) { return corrs[i].measurement2.x() / corrs[i].measurement2.z(); };
    auto y_ = [&corrs](size_t i) { return corrs[i].measurement2.y() / corrs[i].measurement2.z(); };

    for (size_t i = 0, j = 0; i < corrs.size(); i++)
    {
        if (inliers[i])
        {
            P.row(j * 2) << -x(i), -y(i), -1, 0, 0, 0, x(i) * x_(i), y(i) * x_(i), x_(i);
            P.row(j * 2 + 1) << 0, 0, 0, -x(i), -y(i), -1, x(i) * y_(i), y(i) * y_(i), y_(i);
            j++;
        }
    }

    // add constraint that bottom right corner is 1
    P.bottomRows<1>() << 0, 0, 0, 0, 0, 0, 0, 0, 1;
    Eigen::Matrix<double, Eigen::Dynamic, 1> rhs(num_inliers * 2 + 1, 1);
    rhs.setZero();
    rhs.bottomRows<1>() << 1;

    Eigen::Matrix<double, 9, 1> H_ = P.fullPivLu().solve(rhs);
    homography.row(0) = H_.topRows<3>().transpose();
    homography.row(1) = H_.middleRows<3>(3).transpose();
    homography.row(2) = H_.bottomRows<3>().transpose();
    homography /= homography(2, 2); // renormalize in case that constraint wasn't enough
}

double homography_model::error(const correspondence &corr)
{
    return ((homography * corr.measurement1.hnormalized().homogeneous()).hnormalized() -
            corr.measurement2.hnormalized())
        .norm();
}
size_t homography_model::evaluate(const std::vector<correspondence> &corrs, std::vector<bool> &inliers)
{
    size_t num_inliers = 0;
    for (size_t i = 0; i < inliers.size(); i++)
    {
        bool in = error(corrs[i]) < inlier_threshold;
        inliers[i] = in;
        num_inliers += in;
    }
    return num_inliers;
}

template <int n> double fast_pow(double d);

template <> inline double fast_pow<4>(double d)
{
    return (d * d) * (d * d);
}

template <typename Model>
double ransac(const std::vector<correspondence> &matches, Model &model, std::vector<bool> &inliers)
{

    const size_t MIN_ITERATIONS = 10;
    const size_t MAX_ITERATIONS = 500;
    const size_t MAX_INNER_ITERATIONS = 5;
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

    auto random_k_from_n = [&generator](int n) {
        std::array<size_t, Model::MINIMUM_POINTS> indices;
        std::uniform_int_distribution<int> distribution(0, n);
        for (size_t j = 0; j < Model::MINIMUM_POINTS; j++)
        {
            int candidate = distribution(generator);
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

    int probability_iterations = MAX_ITERATIONS;

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

} // namespace opencalibration