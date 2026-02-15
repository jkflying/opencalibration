#pragma once

#include <ankerl/unordered_dense.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace opencalibration::orthomosaic
{

struct RadiometricParams
{
    std::array<double, 3> lab_offset = {0, 0, 0};
    double brdf_coeff = 0;
    std::array<double, 2> slope = {0, 0}; // directional brightness: slope_x, slope_y in image space
};

struct VignettingParams
{
    std::array<double, 3> coeffs = {0, 0, 0}; // polynomial in r^2: c0*r^2 + c1*r^4 + c2*r^6
};

struct ColorCorrespondence
{
    std::array<float, 3> lab_a;
    std::array<float, 3> lab_b;

    size_t camera_id_a;
    size_t camera_id_b;

    uint32_t model_id_a;
    uint32_t model_id_b;

    // Normalized radius in source image (0 = center, 1 = corner) for vignetting
    float normalized_radius_a;
    float normalized_radius_b;

    // View angle (radians) for BRDF correction
    float view_angle_a;
    float view_angle_b;

    // Normalized pixel position in source image, [-1,1] from center
    float normalized_x_a = 0;
    float normalized_y_a = 0;
    float normalized_x_b = 0;
    float normalized_y_b = 0;
};

struct ColorBalanceResult
{
    ankerl::unordered_dense::map<size_t, RadiometricParams> per_image_params;
    ankerl::unordered_dense::map<uint32_t, VignettingParams> per_model_params;
    bool success = false;
    double final_cost = 0;
    int num_iterations = 0;
};

struct CameraPosition
{
    double x, y;
};

ColorBalanceResult solveColorBalance(const std::vector<ColorCorrespondence> &correspondences,
                                     const ankerl::unordered_dense::map<size_t, CameraPosition> &camera_positions = {});

} // namespace opencalibration::orthomosaic
