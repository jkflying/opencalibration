#include <opencalibration/ortho/color_balance.hpp>
#include <opencalibration/ortho/radiometric_cost.hpp>

#include <ceres/autodiff_cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>

#include <spdlog/spdlog.h>

#include <eigen3/Eigen/Dense>

#include <limits>
#include <set>
#include <thread>

namespace opencalibration::orthomosaic
{

namespace
{
constexpr double LAB_MATCH_HUBER_SCALE = 5.0;
constexpr double PRIOR_WEIGHT_PER_SQRT_CORRESPONDENCE = 0.1;

template <int N> void addZeroPrior(ceres::Problem &problem, double *params, double weight)
{
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ZeroPrior<N>, N, N>(new ZeroPrior<N>(weight)), nullptr,
                             params);
}
} // namespace

ColorBalanceResult solveColorBalance(const std::vector<ColorCorrespondence> &correspondences,
                                     const ankerl::unordered_dense::map<size_t, CameraPosition> &camera_positions)
{
    ColorBalanceResult result;

    if (correspondences.empty())
    {
        spdlog::warn("Color balance: no correspondences to solve");
        result.success = false;
        return result;
    }

    std::set<size_t> camera_ids;
    std::set<uint32_t> model_ids;
    for (const auto &corr : correspondences)
    {
        camera_ids.insert(corr.camera_id_a);
        camera_ids.insert(corr.camera_id_b);
        model_ids.insert(corr.model_id_a);
        model_ids.insert(corr.model_id_b);
    }

    spdlog::info("Color balance: {} correspondences, {} cameras, {} camera models", correspondences.size(),
                 camera_ids.size(), model_ids.size());

    for (size_t cam_id : camera_ids)
    {
        result.per_image_params[cam_id] = RadiometricParams{};
    }

    for (uint32_t model_id : model_ids)
    {
        result.per_model_params[model_id] = VignettingParams{};
    }

    ceres::Problem problem;
    auto *view_dir_gain = result.horizontal_view_dir_log_cbrt_gain.data();

    for (const auto &corr : correspondences)
    {
        auto &a = result.per_image_params[corr.camera_id_a];
        auto &b = result.per_image_params[corr.camera_id_b];
        auto *vig_a = result.per_model_params[corr.model_id_a].log_cbrt_falloff_coeffs.data();
        auto *vig_b = result.per_model_params[corr.model_id_b].log_cbrt_falloff_coeffs.data();

        if (corr.model_id_a == corr.model_id_b)
        {
            auto *cost =
                new ceres::AutoDiffCostFunction<RadiometricMatchCostSharedVig, 3, 1, 2, 1, 2, 1, 2, 1, 2, 3, 2>(
                    new RadiometricMatchCostSharedVig(corr));
            problem.AddResidualBlock(cost, new ceres::HuberLoss(LAB_MATCH_HUBER_SCALE), &a.log_cbrt_exposure,
                                     a.ab_offset.data(), &a.brdf_coeff, a.slope.data(), &b.log_cbrt_exposure,
                                     b.ab_offset.data(), &b.brdf_coeff, b.slope.data(), vig_a, view_dir_gain);
        }
        else
        {
            auto *cost = new ceres::AutoDiffCostFunction<RadiometricMatchCost, 3, 1, 2, 1, 2, 3, 1, 2, 1, 2, 3, 2>(
                new RadiometricMatchCost(corr));
            problem.AddResidualBlock(cost, new ceres::HuberLoss(LAB_MATCH_HUBER_SCALE), &a.log_cbrt_exposure,
                                     a.ab_offset.data(), &a.brdf_coeff, a.slope.data(), vig_a, &b.log_cbrt_exposure,
                                     b.ab_offset.data(), &b.brdf_coeff, b.slope.data(), vig_b, view_dir_gain);
        }
    }

    std::unordered_map<size_t, int> cam_corr_counts;
    std::unordered_map<uint32_t, int> model_corr_counts;
    for (const auto &corr : correspondences)
    {
        cam_corr_counts[corr.camera_id_a]++;
        cam_corr_counts[corr.camera_id_b]++;
        model_corr_counts[corr.model_id_a]++;
        model_corr_counts[corr.model_id_b]++;
    }

    for (auto &[cam_id, params] : result.per_image_params)
    {
        double ab_weight =
            PRIOR_WEIGHT_PER_SQRT_CORRESPONDENCE * std::sqrt(static_cast<double>(std::max(1, cam_corr_counts[cam_id])));
        double log_cbrt_weight = ab_weight * L_UNITS_PER_LOG_CBRT_GAIN;
        addZeroPrior<1>(problem, &params.log_cbrt_exposure, log_cbrt_weight);
        addZeroPrior<2>(problem, params.ab_offset.data(), ab_weight);
        addZeroPrior<1>(problem, &params.brdf_coeff, log_cbrt_weight);
        addZeroPrior<2>(problem, params.slope.data(), log_cbrt_weight);
    }

    for (auto &[model_id, vig] : result.per_model_params)
    {
        double scale = std::sqrt(static_cast<double>(std::max(1, model_corr_counts[model_id])));
        addZeroPrior<3>(problem, vig.log_cbrt_falloff_coeffs.data(), PRIOR_WEIGHT_PER_SQRT_CORRESPONDENCE * scale);
    }

    addZeroPrior<2>(problem, view_dir_gain,
                    PRIOR_WEIGHT_PER_SQRT_CORRESPONDENCE * std::sqrt(static_cast<double>(correspondences.size())));

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    options.max_num_iterations = 20;
    options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    options.function_tolerance = 1e-4;
    options.gradient_tolerance = 1e-6;
    options.parameter_tolerance = 1e-4;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    result.success =
        (summary.termination_type == ceres::CONVERGENCE || summary.termination_type == ceres::NO_CONVERGENCE);
    result.final_cost = summary.final_cost;
    result.num_iterations = summary.iterations.size();

    spdlog::info("Color balance: {} after {} iterations, final cost: {:.4f}",
                 summary.termination_type == ceres::CONVERGENCE ? "converged" : "did not converge",
                 result.num_iterations, result.final_cost);

    // Remove gauge freedom: fit a plane offset = a*x + b*y + c to the solved offsets
    // via SVD least squares and subtract it. This removes both constant bias and any
    // linear spatial gradient that the relative-only match costs allow to develop.
    if (!camera_positions.empty())
    {
        std::vector<size_t> cam_order;
        cam_order.reserve(result.per_image_params.size());
        for (const auto &[cam_id, params] : result.per_image_params)
        {
            if (camera_positions.count(cam_id))
                cam_order.push_back(cam_id);
        }

        if (cam_order.size() >= 3)
        {
            int n = static_cast<int>(cam_order.size());

            // A = [x, y, 1] for each camera
            Eigen::MatrixXd A(n, 3);
            for (int i = 0; i < n; i++)
            {
                const auto &pos = camera_positions.at(cam_order[i]);
                A(i, 0) = pos.x;
                A(i, 1) = pos.y;
                A(i, 2) = 1.0;
            }

            // SVD decomposition of A (computed once, reused for all channels)
            auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

            auto offsetOfChannel = [](RadiometricParams &p, int c) -> double & {
                return c == 0 ? p.log_cbrt_exposure : p.ab_offset[c - 1];
            };

            for (int c = 0; c < 3; c++)
            {
                Eigen::VectorXd b(n);
                for (int i = 0; i < n; i++)
                {
                    b(i) = offsetOfChannel(result.per_image_params[cam_order[i]], c);
                }

                // Solve A * [a, b, c]^T = offsets in least squares
                Eigen::Vector3d plane = svd.solve(b);

                spdlog::info("Color balance: channel {} plane fit: {:.4f}*x + {:.4f}*y + {:.4f}", c, plane(0), plane(1),
                             plane(2));

                for (int i = 0; i < n; i++)
                {
                    const auto &pos = camera_positions.at(cam_order[i]);
                    double fitted = plane(0) * pos.x + plane(1) * pos.y + plane(2);
                    offsetOfChannel(result.per_image_params[cam_order[i]], c) -= fitted;
                }
            }
        }
    }

    double max_log_cbrt_exposure = 0;
    double max_slope = 0;
    for (const auto &[cam_id, params] : result.per_image_params)
    {
        max_log_cbrt_exposure = std::max(max_log_cbrt_exposure, std::abs(params.log_cbrt_exposure));
        max_slope = std::max(max_slope, std::max(std::abs(params.slope[0]), std::abs(params.slope[1])));
    }
    spdlog::info("Color balance: max log-cbrt exposure after detrending: {:.4f}, max slope: {:.4f}",
                 max_log_cbrt_exposure, max_slope);

    return result;
}

} // namespace opencalibration::orthomosaic
