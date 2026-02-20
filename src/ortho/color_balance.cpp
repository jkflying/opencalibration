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

    for (const auto &corr : correspondences)
    {
        auto &params_a = result.per_image_params[corr.camera_id_a];
        auto &params_b = result.per_image_params[corr.camera_id_b];
        auto &vig_a = result.per_model_params[corr.model_id_a];
        auto &vig_b = result.per_model_params[corr.model_id_b];

        if (corr.model_id_a == corr.model_id_b)
        {
            // Same model: use shared vignetting variant to avoid Ceres duplicate parameter block error
            using CostType = RadiometricMatchCostSharedVig;
            auto *cost = new ceres::AutoDiffCostFunction<CostType, CostType::NUM_RESIDUALS, CostType::NUM_PARAMETERS_1,
                                                         CostType::NUM_PARAMETERS_2, CostType::NUM_PARAMETERS_3,
                                                         CostType::NUM_PARAMETERS_4, CostType::NUM_PARAMETERS_5,
                                                         CostType::NUM_PARAMETERS_6, CostType::NUM_PARAMETERS_7>(
                new CostType(corr.lab_a.data(), corr.lab_b.data(), corr.normalized_radius_a, corr.normalized_radius_b,
                             corr.view_angle_a, corr.view_angle_b, corr.normalized_x_a, corr.normalized_y_a,
                             corr.normalized_x_b, corr.normalized_y_b));

            problem.AddResidualBlock(cost, new ceres::HuberLoss(5.0), params_a.lab_offset.data(), &params_a.brdf_coeff,
                                     params_b.lab_offset.data(), &params_b.brdf_coeff, vig_a.coeffs.data(),
                                     params_a.slope.data(), params_b.slope.data());
        }
        else
        {
            using CostType = RadiometricMatchCost;
            auto *cost = new ceres::AutoDiffCostFunction<
                CostType, CostType::NUM_RESIDUALS, CostType::NUM_PARAMETERS_1, CostType::NUM_PARAMETERS_2,
                CostType::NUM_PARAMETERS_3, CostType::NUM_PARAMETERS_4, CostType::NUM_PARAMETERS_5,
                CostType::NUM_PARAMETERS_6, CostType::NUM_PARAMETERS_7, CostType::NUM_PARAMETERS_8>(
                new CostType(corr.lab_a.data(), corr.lab_b.data(), corr.normalized_radius_a, corr.normalized_radius_b,
                             corr.view_angle_a, corr.view_angle_b, corr.normalized_x_a, corr.normalized_y_a,
                             corr.normalized_x_b, corr.normalized_y_b));

            problem.AddResidualBlock(cost, new ceres::HuberLoss(5.0), params_a.lab_offset.data(), &params_a.brdf_coeff,
                                     vig_a.coeffs.data(), params_b.lab_offset.data(), &params_b.brdf_coeff,
                                     vig_b.coeffs.data(), params_a.slope.data(), params_b.slope.data());
        }
    }

    std::unordered_map<size_t, int> cam_corr_counts;
    for (const auto &corr : correspondences)
    {
        cam_corr_counts[corr.camera_id_a]++;
        cam_corr_counts[corr.camera_id_b]++;
    }

    for (auto &[cam_id, params] : result.per_image_params)
    {
        double scale = std::sqrt(static_cast<double>(std::max(1, cam_corr_counts[cam_id])));

        auto *exposure_cost =
            new ceres::AutoDiffCostFunction<ExposurePrior, ExposurePrior::NUM_RESIDUALS,
                                            ExposurePrior::NUM_PARAMETERS_1>(new ExposurePrior(0.1 * scale));
        problem.AddResidualBlock(exposure_cost, nullptr, params.lab_offset.data());

        auto *brdf_cost =
            new ceres::AutoDiffCostFunction<BRDFPrior, BRDFPrior::NUM_RESIDUALS, BRDFPrior::NUM_PARAMETERS_1>(
                new BRDFPrior(0.1 * scale));
        problem.AddResidualBlock(brdf_cost, nullptr, &params.brdf_coeff);
    }

    std::unordered_map<uint32_t, int> model_corr_counts;
    for (const auto &corr : correspondences)
    {
        model_corr_counts[corr.model_id_a]++;
        model_corr_counts[corr.model_id_b]++;
    }

    for (auto &[model_id, vig] : result.per_model_params)
    {
        double scale = std::sqrt(static_cast<double>(std::max(1, model_corr_counts[model_id])));
        auto *vig_cost =
            new ceres::AutoDiffCostFunction<VignettingPrior, VignettingPrior::NUM_RESIDUALS,
                                            VignettingPrior::NUM_PARAMETERS_1>(new VignettingPrior(0.1 * scale));
        problem.AddResidualBlock(vig_cost, nullptr, vig.coeffs.data());
    }

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

            for (int c = 0; c < 3; c++)
            {
                // b = offset values for channel c
                Eigen::VectorXd b(n);
                for (int i = 0; i < n; i++)
                {
                    b(i) = result.per_image_params[cam_order[i]].lab_offset[c];
                }

                // Solve A * [a, b, c]^T = offsets in least squares
                Eigen::Vector3d plane = svd.solve(b);

                spdlog::info("Color balance: channel {} plane fit: {:.4f}*x + {:.4f}*y + {:.4f}", c, plane(0), plane(1),
                             plane(2));

                for (int i = 0; i < n; i++)
                {
                    const auto &pos = camera_positions.at(cam_order[i]);
                    double fitted = plane(0) * pos.x + plane(1) * pos.y + plane(2);
                    result.per_image_params[cam_order[i]].lab_offset[c] -= fitted;
                }
            }
        }
    }

    double max_L_offset = 0;
    double max_slope = 0;
    for (const auto &[cam_id, params] : result.per_image_params)
    {
        max_L_offset = std::max(max_L_offset, std::abs(params.lab_offset[0]));
        max_slope = std::max(max_slope, std::max(std::abs(params.slope[0]), std::abs(params.slope[1])));
    }
    spdlog::info("Color balance: max L offset after detrending: {:.2f}, max slope: {:.2f}", max_L_offset, max_slope);

    return result;
}

} // namespace opencalibration::orthomosaic
