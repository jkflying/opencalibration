#pragma once

#include <ankerl/unordered_dense.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace opencalibration::orthomosaic
{

struct RadiometricParams
{
    double log_cbrt_exposure = 0;
    std::array<double, 2> ab_offset = {0, 0};
    std::array<double, 2> slope = {0, 0};
};

struct VignettingParams
{
    std::array<double, 3> log_cbrt_falloff_coeffs = {0, 0, 0};
};

struct SampleGeometry
{
    float normalized_radius = 0;
    float normalized_x = 0;
    float normalized_y = 0;
    float view_dir_x = 0;
    float view_dir_y = 0;
};

struct ColorCorrespondence
{
    std::array<float, 3> lab_a;
    std::array<float, 3> lab_b;

    size_t camera_id_a;
    size_t camera_id_b;

    uint32_t model_id_a;
    uint32_t model_id_b;

    SampleGeometry geometry_a;
    SampleGeometry geometry_b;
};

struct ColorBalanceResult
{
    ankerl::unordered_dense::map<size_t, RadiometricParams> per_image_params;
    ankerl::unordered_dense::map<uint32_t, VignettingParams> per_model_params;
    std::array<double, 2> horizontal_view_dir_log_cbrt_gain = {0, 0};
    bool success = false;
    double final_cost = 0;
    int num_iterations = 0;
};

ColorBalanceResult solveColorBalance(const std::vector<ColorCorrespondence> &correspondences,
                                     const ankerl::unordered_dense::map<size_t, double> &exif_exposure_values = {});

} // namespace opencalibration::orthomosaic
