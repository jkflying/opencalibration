#include <opencalibration/ortho/color_balance.hpp>
#include <opencalibration/ortho/radiometric_cost.hpp>

#include <opencalibration/relax/padded_autodiff_cost_function.hpp>
#include <ceres/loss_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <thread>

namespace opencalibration::orthomosaic
{

namespace
{
constexpr double LAB_MATCH_HUBER_SCALE = 5.0;
constexpr double PRIOR_WEIGHT_PER_SQRT_CORRESPONDENCE = 0.1;
constexpr double OFFSET_GAUGE_PRIOR_FRACTION = 0.01;
constexpr double EXIF_EXPOSURE_PRIOR_FRACTION = 1.0;
constexpr double VIEW_DIR_PRIOR_FRACTION = 1e-3;

template <int N>
void addPrior(ceres::Problem &problem, double *params, double weight, const std::array<double, N> &target = {})
{
    problem.AddResidualBlock(new PaddedAutoDiffCostFunction<TargetPrior<N>, N, N>(new TargetPrior<N>{weight, target}),
                             nullptr, params);
}

double exposureValueOf(const ankerl::unordered_dense::map<size_t, double> &exif_exposure_values, size_t camera_id)
{
    const auto it = exif_exposure_values.find(camera_id);
    return it == exif_exposure_values.end() ? std::numeric_limits<double>::quiet_NaN() : it->second;
}

double priorWeight(size_t num_correspondences)
{
    return PRIOR_WEIGHT_PER_SQRT_CORRESPONDENCE * std::sqrt(static_cast<double>(num_correspondences));
}

double median(std::vector<double> values)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

} // namespace

ColorBalanceResult solveColorBalance(const std::vector<ColorCorrespondence> &correspondences,
                                     const ankerl::unordered_dense::map<size_t, double> &exif_exposure_values)
{
    ColorBalanceResult result;

    if (correspondences.empty())
    {
        spdlog::warn("Color balance: no correspondences to solve");
        result.success = false;
        return result;
    }

    std::map<size_t, size_t> camera_correspondences;
    std::map<uint32_t, size_t> model_correspondences;
    for (const auto &corr : correspondences)
    {
        camera_correspondences[corr.camera_id_a]++;
        camera_correspondences[corr.camera_id_b]++;
        model_correspondences[corr.model_id_a]++;
        model_correspondences[corr.model_id_b]++;
    }

    spdlog::info("Color balance: {} correspondences, {} cameras, {} camera models", correspondences.size(),
                 camera_correspondences.size(), model_correspondences.size());

    std::vector<double> exposure_values;
    for (const auto &[cam_id, count] : camera_correspondences)
    {
        result.per_image_params[cam_id] = RadiometricParams{};
        const double exposure_value = exposureValueOf(exif_exposure_values, cam_id);
        if (exposure_value > 0)
            exposure_values.push_back(exposure_value);
    }
    for (const auto &[model_id, count] : model_correspondences)
        result.per_model_params[model_id] = VignettingParams{};

    spdlog::info("Color balance: {} of {} cameras have EXIF exposure", exposure_values.size(),
                 camera_correspondences.size());

    const double median_exposure_value = median(exposure_values);

    ceres::Problem problem;
    double *view_dir_gain = result.horizontal_view_dir_log_cbrt_gain.data();

    for (const auto &corr : correspondences)
    {
        auto &a = result.per_image_params[corr.camera_id_a];
        auto &b = result.per_image_params[corr.camera_id_b];
        auto *vig_a = result.per_model_params[corr.model_id_a].log_cbrt_falloff_coeffs.data();
        auto *vig_b = result.per_model_params[corr.model_id_b].log_cbrt_falloff_coeffs.data();

        if (corr.model_id_a == corr.model_id_b)
        {
            auto *cost = new PaddedAutoDiffCostFunction<RadiometricMatchCostSharedVig, 3, 1, 2, 2, 1, 2, 2, 3, 2>(
                new RadiometricMatchCostSharedVig(corr));
            problem.AddResidualBlock(cost, new ceres::HuberLoss(LAB_MATCH_HUBER_SCALE), &a.log_cbrt_exposure,
                                     a.ab_offset.data(), a.slope.data(), &b.log_cbrt_exposure, b.ab_offset.data(),
                                     b.slope.data(), vig_a, view_dir_gain);
        }
        else
        {
            auto *cost = new PaddedAutoDiffCostFunction<RadiometricMatchCost, 3, 1, 2, 2, 3, 1, 2, 2, 3, 2>(
                new RadiometricMatchCost(corr));
            problem.AddResidualBlock(cost, new ceres::HuberLoss(LAB_MATCH_HUBER_SCALE), &a.log_cbrt_exposure,
                                     a.ab_offset.data(), a.slope.data(), vig_a, &b.log_cbrt_exposure,
                                     b.ab_offset.data(), b.slope.data(), vig_b, view_dir_gain);
        }
    }

    for (auto &[cam_id, params] : result.per_image_params)
    {
        const double weight = priorWeight(camera_correspondences.at(cam_id));
        const double log_cbrt_weight = weight * L_UNITS_PER_LOG_CBRT_GAIN;
        const double exposure_value = exposureValueOf(exif_exposure_values, cam_id);
        if (exposure_value > 0)
        {
            params.log_cbrt_exposure = std::log(std::cbrt(exposure_value / median_exposure_value));
            addPrior<1>(problem, &params.log_cbrt_exposure, EXIF_EXPOSURE_PRIOR_FRACTION * log_cbrt_weight,
                        {params.log_cbrt_exposure});
        }
        else
            addPrior<1>(problem, &params.log_cbrt_exposure, OFFSET_GAUGE_PRIOR_FRACTION * log_cbrt_weight);
        addPrior<2>(problem, params.ab_offset.data(), OFFSET_GAUGE_PRIOR_FRACTION * weight);
        addPrior<2>(problem, params.slope.data(), log_cbrt_weight);
    }

    for (auto &[model_id, vig] : result.per_model_params)
        addPrior<3>(problem, vig.log_cbrt_falloff_coeffs.data(), priorWeight(model_correspondences.at(model_id)));

    addPrior<2>(problem, view_dir_gain, VIEW_DIR_PRIOR_FRACTION * priorWeight(correspondences.size()));

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    options.max_num_iterations = 30;
    options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    options.function_tolerance = 1e-5;
    options.gradient_tolerance = 1e-8;
    options.parameter_tolerance = 1e-5;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    result.success =
        (summary.termination_type == ceres::CONVERGENCE || summary.termination_type == ceres::NO_CONVERGENCE);
    result.final_cost = summary.final_cost;
    result.num_iterations = summary.iterations.size();

    spdlog::info("Color balance: {} after {} iterations, final cost: {:.4f}, horizontal view gain ({:.4f}, {:.4f})",
                 summary.termination_type == ceres::CONVERGENCE ? "converged" : "did not converge",
                 result.num_iterations, result.final_cost, view_dir_gain[0], view_dir_gain[1]);

    return result;
}

} // namespace opencalibration::orthomosaic
