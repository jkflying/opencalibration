#pragma once

#include <cmath>

namespace opencalibration::orthomosaic
{

// Ceres cost function: penalize difference between two cameras observing the same point
// after applying per-image radiometric corrections.
//
// Radiometric model (additive in LAB):
//   corrected_L = observed_L - offset_L - (vig[0]*r^2 + vig[1]*r^4 + vig[2]*r^6)
//                 - brdf*theta^2 - slope[0]*nx - slope[1]*ny
//   corrected_a = observed_a - offset_a
//   corrected_b = observed_b - offset_b
//
// Residual = corrected_a - corrected_b (should be zero for matching points)
//
// Parameter blocks: lab_offset_a[3], brdf_a[1], vig_a[3], lab_offset_b[3], brdf_b[1], vig_b[3],
//                   slope_a[2], slope_b[2]
struct RadiometricMatchCost
{
    static constexpr int NUM_RESIDUALS = 3;
    static constexpr int NUM_PARAMETERS_1 = 3; // lab_offset_a
    static constexpr int NUM_PARAMETERS_2 = 1; // brdf_a
    static constexpr int NUM_PARAMETERS_3 = 3; // vig_a
    static constexpr int NUM_PARAMETERS_4 = 3; // lab_offset_b
    static constexpr int NUM_PARAMETERS_5 = 1; // brdf_b
    static constexpr int NUM_PARAMETERS_6 = 3; // vig_b
    static constexpr int NUM_PARAMETERS_7 = 2; // slope_a
    static constexpr int NUM_PARAMETERS_8 = 2; // slope_b

    float _observed_a[3];
    float _observed_b[3];
    float _r_a;         // normalized radius in image A
    float _r_b;         // normalized radius in image B
    float _theta_a;     // view angle for image A
    float _theta_b;     // view angle for image B
    float _nx_a, _ny_a; // normalized pixel position in image A
    float _nx_b, _ny_b; // normalized pixel position in image B

    RadiometricMatchCost(const float observed_a[3], const float observed_b[3], float r_a, float r_b, float theta_a,
                         float theta_b, float nx_a, float ny_a, float nx_b, float ny_b)
        : _r_a(r_a), _r_b(r_b), _theta_a(theta_a), _theta_b(theta_b), _nx_a(nx_a), _ny_a(ny_a), _nx_b(nx_b), _ny_b(ny_b)
    {
        for (int i = 0; i < 3; i++)
        {
            _observed_a[i] = observed_a[i];
            _observed_b[i] = observed_b[i];
        }
    }

    template <typename T>
    bool operator()(const T *lab_offset_a, const T *brdf_a, const T *vig_a, const T *lab_offset_b, const T *brdf_b,
                    const T *vig_b, const T *slope_a, const T *slope_b, T *residuals) const
    {
        T r2_a = T(_r_a * _r_a);
        T vig_corr_a = vig_a[0] * r2_a + vig_a[1] * r2_a * r2_a + vig_a[2] * r2_a * r2_a * r2_a;

        T r2_b = T(_r_b * _r_b);
        T vig_corr_b = vig_b[0] * r2_b + vig_b[1] * r2_b * r2_b + vig_b[2] * r2_b * r2_b * r2_b;

        T brdf_corr_a = brdf_a[0] * T(_theta_a * _theta_a);
        T brdf_corr_b = brdf_b[0] * T(_theta_b * _theta_b);

        T slope_corr_a = slope_a[0] * T(_nx_a) + slope_a[1] * T(_ny_a);
        T slope_corr_b = slope_b[0] * T(_nx_b) + slope_b[1] * T(_ny_b);

        for (int c = 0; c < 3; c++)
        {
            T corr_a = T(_observed_a[c]) - lab_offset_a[c];
            T corr_b = T(_observed_b[c]) - lab_offset_b[c];
            if (c == 0)
            {
                // L channel gets vignetting + BRDF + slope correction
                corr_a -= vig_corr_a + brdf_corr_a + slope_corr_a;
                corr_b -= vig_corr_b + brdf_corr_b + slope_corr_b;
            }
            residuals[c] = corr_a - corr_b;
        }
        return true;
    }
};

// Same as RadiometricMatchCost but for when both cameras share the same camera model
// (so vignetting parameter block is shared, avoiding Ceres duplicate parameter error)
// Parameter blocks: lab_offset_a[3], brdf_a[1], lab_offset_b[3], brdf_b[1], vig_shared[3],
//                   slope_a[2], slope_b[2]
struct RadiometricMatchCostSharedVig
{
    static constexpr int NUM_RESIDUALS = 3;
    static constexpr int NUM_PARAMETERS_1 = 3; // lab_offset_a
    static constexpr int NUM_PARAMETERS_2 = 1; // brdf_a
    static constexpr int NUM_PARAMETERS_3 = 3; // lab_offset_b
    static constexpr int NUM_PARAMETERS_4 = 1; // brdf_b
    static constexpr int NUM_PARAMETERS_5 = 3; // vig_shared
    static constexpr int NUM_PARAMETERS_6 = 2; // slope_a
    static constexpr int NUM_PARAMETERS_7 = 2; // slope_b

    float _observed_a[3];
    float _observed_b[3];
    float _r_a, _r_b;
    float _theta_a, _theta_b;
    float _nx_a, _ny_a;
    float _nx_b, _ny_b;

    RadiometricMatchCostSharedVig(const float observed_a[3], const float observed_b[3], float r_a, float r_b,
                                  float theta_a, float theta_b, float nx_a, float ny_a, float nx_b, float ny_b)
        : _r_a(r_a), _r_b(r_b), _theta_a(theta_a), _theta_b(theta_b), _nx_a(nx_a), _ny_a(ny_a), _nx_b(nx_b), _ny_b(ny_b)
    {
        for (int i = 0; i < 3; i++)
        {
            _observed_a[i] = observed_a[i];
            _observed_b[i] = observed_b[i];
        }
    }

    template <typename T>
    bool operator()(const T *lab_offset_a, const T *brdf_a, const T *lab_offset_b, const T *brdf_b, const T *vig,
                    const T *slope_a, const T *slope_b, T *residuals) const
    {
        T r2_a = T(_r_a * _r_a);
        T vig_corr_a = vig[0] * r2_a + vig[1] * r2_a * r2_a + vig[2] * r2_a * r2_a * r2_a;

        T r2_b = T(_r_b * _r_b);
        T vig_corr_b = vig[0] * r2_b + vig[1] * r2_b * r2_b + vig[2] * r2_b * r2_b * r2_b;

        T brdf_corr_a = brdf_a[0] * T(_theta_a * _theta_a);
        T brdf_corr_b = brdf_b[0] * T(_theta_b * _theta_b);

        T slope_corr_a = slope_a[0] * T(_nx_a) + slope_a[1] * T(_ny_a);
        T slope_corr_b = slope_b[0] * T(_nx_b) + slope_b[1] * T(_ny_b);

        for (int c = 0; c < 3; c++)
        {
            T corr_a = T(_observed_a[c]) - lab_offset_a[c];
            T corr_b = T(_observed_b[c]) - lab_offset_b[c];
            if (c == 0)
            {
                corr_a -= vig_corr_a + brdf_corr_a + slope_corr_a;
                corr_b -= vig_corr_b + brdf_corr_b + slope_corr_b;
            }
            residuals[c] = corr_a - corr_b;
        }
        return true;
    }
};

// Prior: penalize per-image LAB offset away from zero
struct ExposurePrior
{
    static constexpr int NUM_RESIDUALS = 3;
    static constexpr int NUM_PARAMETERS_1 = 3; // lab_offset

    double _weight;

    explicit ExposurePrior(double weight) : _weight(weight)
    {
    }

    template <typename T> bool operator()(const T *lab_offset, T *residuals) const
    {
        for (int c = 0; c < 3; c++)
        {
            residuals[c] = T(_weight) * lab_offset[c];
        }
        return true;
    }
};

// Prior: penalize vignetting coefficients away from zero
struct VignettingPrior
{
    static constexpr int NUM_RESIDUALS = 3;
    static constexpr int NUM_PARAMETERS_1 = 3; // vignetting coeffs

    double _weight;

    explicit VignettingPrior(double weight) : _weight(weight)
    {
    }

    template <typename T> bool operator()(const T *vig, T *residuals) const
    {
        for (int c = 0; c < 3; c++)
        {
            residuals[c] = T(_weight) * vig[c];
        }
        return true;
    }
};

// Prior: penalize BRDF coefficient away from zero
struct BRDFPrior
{
    static constexpr int NUM_RESIDUALS = 1;
    static constexpr int NUM_PARAMETERS_1 = 1; // brdf coeff

    double _weight;

    explicit BRDFPrior(double weight) : _weight(weight)
    {
    }

    template <typename T> bool operator()(const T *brdf, T *residuals) const
    {
        residuals[0] = T(_weight) * brdf[0];
        return true;
    }
};

} // namespace opencalibration::orthomosaic
