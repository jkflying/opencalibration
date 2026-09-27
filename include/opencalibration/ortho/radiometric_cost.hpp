#pragma once

#include <opencalibration/ortho/color_balance.hpp>

#include <cmath>

namespace opencalibration::orthomosaic
{

constexpr double LAB_L_BLACK_OFFSET = 16.0;
constexpr double LAB_L_MID_GREY = 50.0;
constexpr double L_UNITS_PER_LOG_CBRT_GAIN = LAB_L_MID_GREY + LAB_L_BLACK_OFFSET;

template <typename T> void scaleLabByCubeRootOfLinearGain(const T lab[3], T cbrt_linear_gain, T out[3])
{
    out[0] = (lab[0] + T(LAB_L_BLACK_OFFSET)) * cbrt_linear_gain - T(LAB_L_BLACK_OFFSET);
    out[1] = lab[1] * cbrt_linear_gain;
    out[2] = lab[2] * cbrt_linear_gain;
}

template <typename T> T vignettingLogCbrtFalloff(const T *log_cbrt_falloff_coeffs, float normalized_radius)
{
    T r2 = T(normalized_radius * normalized_radius);
    return log_cbrt_falloff_coeffs[0] * r2 + log_cbrt_falloff_coeffs[1] * r2 * r2 +
           log_cbrt_falloff_coeffs[2] * r2 * r2 * r2;
}

template <typename T>
void removeVignetting(const T lab[3], const T *log_cbrt_falloff_coeffs, float normalized_radius, T out[3])
{
    T cbrt_inverse_falloff = exp(-vignettingLogCbrtFalloff(log_cbrt_falloff_coeffs, normalized_radius));
    scaleLabByCubeRootOfLinearGain(lab, cbrt_inverse_falloff, out);
}

template <typename T> struct RadiometricModel
{
    const T *log_cbrt_exposure;
    const T *ab_offset;
    const T *brdf;
    const T *slope;
    const T *vignetting;
    const T *horizontal_view_dir_gain;
};

template <typename T> T totalLogCbrtGain(const RadiometricModel<T> &m, const SampleGeometry &g)
{
    return vignettingLogCbrtFalloff(m.vignetting, g.normalized_radius) + m.log_cbrt_exposure[0] +
           m.brdf[0] * T(g.view_angle_rad * g.view_angle_rad) + m.slope[0] * T(g.normalized_x) +
           m.slope[1] * T(g.normalized_y) + m.horizontal_view_dir_gain[0] * T(g.horizontal_view_dir_x) +
           m.horizontal_view_dir_gain[1] * T(g.horizontal_view_dir_y);
}

template <typename T>
void correctRadiometry(const T lab[3], const RadiometricModel<T> &m, const SampleGeometry &g, T out[3])
{
    scaleLabByCubeRootOfLinearGain(lab, exp(-totalLogCbrtGain(m, g)), out);
    out[1] -= m.ab_offset[0];
    out[2] -= m.ab_offset[1];
}

struct RadiometricMatchCost
{
    std::array<float, 3> _observed_a, _observed_b;
    SampleGeometry _geometry_a, _geometry_b;

    explicit RadiometricMatchCost(const ColorCorrespondence &corr)
        : _observed_a(corr.lab_a), _observed_b(corr.lab_b), _geometry_a(corr.geometry_a), _geometry_b(corr.geometry_b)
    {
    }

    template <typename T>
    bool operator()(const T *exposure_a, const T *ab_a, const T *brdf_a, const T *slope_a, const T *vig_a,
                    const T *exposure_b, const T *ab_b, const T *brdf_b, const T *slope_b, const T *vig_b,
                    const T *view_dir_gain, T *residuals) const
    {
        T obs_a[3] = {T(_observed_a[0]), T(_observed_a[1]), T(_observed_a[2])};
        T obs_b[3] = {T(_observed_b[0]), T(_observed_b[1]), T(_observed_b[2])};
        T corr_a[3], corr_b[3];
        correctRadiometry(obs_a, RadiometricModel<T>{exposure_a, ab_a, brdf_a, slope_a, vig_a, view_dir_gain},
                          _geometry_a, corr_a);
        correctRadiometry(obs_b, RadiometricModel<T>{exposure_b, ab_b, brdf_b, slope_b, vig_b, view_dir_gain},
                          _geometry_b, corr_b);
        T mean_brightness_above_black = T(0.5) * (corr_a[0] + corr_b[0]) + T(LAB_L_BLACK_OFFSET);
        T brightness_invariant_scale = T(L_UNITS_PER_LOG_CBRT_GAIN) / mean_brightness_above_black;
        for (int c = 0; c < 3; c++)
            residuals[c] = (corr_a[c] - corr_b[c]) * brightness_invariant_scale;
        return true;
    }
};

struct RadiometricMatchCostSharedVig
{
    RadiometricMatchCost _cost;

    explicit RadiometricMatchCostSharedVig(const ColorCorrespondence &corr) : _cost(corr)
    {
    }

    template <typename T>
    bool operator()(const T *exposure_a, const T *ab_a, const T *brdf_a, const T *slope_a, const T *exposure_b,
                    const T *ab_b, const T *brdf_b, const T *slope_b, const T *vig, const T *view_dir_gain,
                    T *residuals) const
    {
        return _cost(exposure_a, ab_a, brdf_a, slope_a, vig, exposure_b, ab_b, brdf_b, slope_b, vig, view_dir_gain,
                     residuals);
    }
};

template <int N> struct ZeroPrior
{
    double _weight;

    explicit ZeroPrior(double weight) : _weight(weight)
    {
    }

    template <typename T> bool operator()(const T *params, T *residuals) const
    {
        for (int i = 0; i < N; i++)
            residuals[i] = T(_weight) * params[i];
        return true;
    }
};

} // namespace opencalibration::orthomosaic
