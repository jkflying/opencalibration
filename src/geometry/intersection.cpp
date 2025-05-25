#include <opencalibration/geometry/intersection.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>

#include <Eigen/Geometry>
#include <ceres/jet.h>
#include <ceres/tiny_solver.h>

namespace
{
using namespace opencalibration;

struct pixel_cost_functor
{
    using Scalar = double;
    enum
    {
        NUM_RESIDUALS = 2,
        NUM_PARAMETERS = 3
    };

    pixel_cost_functor(const CameraModel &model, const Eigen::Vector3d &pos, const Eigen::Quaterniond &rot,
                       const Eigen::Vector2d &px)
        : model(model), pos(pos), rot(rot), px(px)
    {
    }

    template <typename T> bool pixel_cost_function(const T *parameters, T *residuals) const
    {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> point(parameters);
        Eigen::Map<Eigen::Matrix<T, 2, 1>> res(residuals);

        res = image_from_3d<T>(rot.inverse().cast<T>() * (point - pos.cast<T>()), model.cast<T>()) - px.cast<T>();

        return true;
    }

    bool operator()(const double *parameters, double *residuals) const
    {
        return pixel_cost_function<double>(parameters, residuals);
    }

    bool operator()(const double *parameters, double *residuals, double *jacobian) const
    {
        using T = ceres::Jet<double, 3>;
        Eigen::Matrix<T, 3, 1> paramsT = Eigen::Map<const Eigen::Vector3d>(parameters).cast<T>();
        for (Eigen::Index i = 0; i < NUM_PARAMETERS; i++)
        {
            paramsT[i].v[i] = 1;
        }

        Eigen::Matrix<T, 2, 1> resT;

        bool ret = pixel_cost_function<T>(paramsT.data(), resT.data());

        Eigen::Map<Eigen::Matrix<double, NUM_RESIDUALS, NUM_PARAMETERS>> jac(jacobian);
        Eigen::Map<Eigen::Vector2d> res(residuals);

        for (Eigen::Index i = 0; i < NUM_RESIDUALS; i++)
        {
            res[i] = resT[i].a;
            jac.row(i) = resT[i].v;
        }

        return ret;
    }

    const CameraModel &model;
    const Eigen::Vector3d &pos;
    const Eigen::Quaterniond &rot;
    const Eigen::Vector2d &px;
};

struct twin_pixel_cost_functor
{
    twin_pixel_cost_functor(pixel_cost_functor &&f1_arg, pixel_cost_functor &&f2_arg) : f1(f1_arg), f2(f2_arg)
    {
    }

    pixel_cost_functor f1, f2;

    using Scalar = pixel_cost_functor::Scalar;
    enum
    {
        NUM_RESIDUALS = pixel_cost_functor::NUM_RESIDUALS * 2,
        NUM_PARAMETERS = pixel_cost_functor::NUM_PARAMETERS
    };

    bool operator()(const double *parameters, double *residuals, double *jacobian) const
    {
        if (jacobian == nullptr)
        {
            return f1(parameters, residuals) && f2(parameters, residuals + pixel_cost_functor::NUM_RESIDUALS);
        }
        else
        {
            Eigen::Matrix<double, 2, 3> jac1, jac2;
            if (f1(parameters, residuals, jac1.data()) &&
                f2(parameters, residuals + pixel_cost_functor::NUM_RESIDUALS, jac2.data()))
            {
                Eigen::Map<Eigen::Matrix<double, 4, 3>> jac(jacobian);
                jac.template topRows<2>() = jac1;
                jac.template bottomRows<2>() = jac2;
                return true;
            }

            return false;
        }
    }
};

} // namespace

namespace opencalibration
{
std::pair<Eigen::Vector3d, double> rayIntersection(const ray_d &r1, const ray_d &r2)
{
    Eigen::Vector3d res{NAN, NAN, NAN};
    double error = NAN;

    double n1dn1 = r1.dir.dot(r1.dir);
    double n1dn2 = r1.dir.dot(r2.dir);
    double n2dn2 = r2.dir.dot(r2.dir);

    auto sqr = [](double d) { return d * d; };
    double scale_denom = sqr(n1dn2) - n1dn1 * n2dn2;

    if (scale_denom < 0.01) // signal to noise ratio is too low
    {

        Eigen::Vector3d offset = (r2.offset - r1.offset);
        double offset1 = offset.dot(r1.dir);
        double offset2 = offset.dot(r2.dir);
        double norm_scale = 1. / scale_denom;

        double t = (offset2 * n1dn2 - offset1 * n2dn2) * norm_scale;
        double s = (offset2 * n1dn1 - offset1 * n1dn2) * norm_scale;

        Eigen::Vector3d p1 = (r1.offset + t * r1.dir), p2 = (r2.offset + s * r2.dir);
        res = 0.5 * (p1 + p2);
        error = (p1 - p2).squaredNorm() * (t >= 0 && s >= 0 ? 1 : -1);
    }

    return std::make_pair(res, error);
}

std::pair<Eigen::Vector3d, double> rayIntersection(const std::vector<ray_d> &rays)
{
    Eigen::Vector3d res{NAN, NAN, NAN};
    double error = NAN;

    if (rays.size() > 1)
    {
        auto init = rayIntersection(rays[0], rays[1]);
        res = init.first;
        error = init.second;

        // TODO: better optimization taking into account multiple rays and a robust cost function
    }

    return std::make_pair(res, error);
}

std::pair<Eigen::Vector3d, double> rayIntersection(const CameraModel &model1, CameraModel &model2,
                                                   const Eigen::Vector3d &pos1, const Eigen::Vector3d &pos2,
                                                   const Eigen::Quaterniond &rot1, const Eigen::Quaterniond &rot2,
                                                   const Eigen::Vector2d &px1, const Eigen::Vector2d &px2)
{
    ray_d ray1{rot1 * image_to_3d(px1, model1), pos1};
    ray_d ray2{rot2 * image_to_3d(px2, model2), pos2};

    std::pair<Eigen::Vector3d, double> initial_guess = rayIntersection(ray1, ray2);

    const twin_pixel_cost_functor func{pixel_cost_functor(model1, pos1, rot1, px1),
                                       pixel_cost_functor(model2, pos2, rot2, px2)};

    ceres::TinySolver<twin_pixel_cost_functor> solver;
    solver.options.max_num_iterations = 50;
    solver.options.cost_threshold = 1e-7;
    solver.options.parameter_tolerance = 1e-14;
    solver.options.gradient_tolerance = 1e-12;
    solver.options.initial_trust_region_radius = 1e6;

    const auto &summary = solver.Solve(func, &initial_guess.first);
    initial_guess.second = summary.final_cost;

    return initial_guess;
}

} // namespace opencalibration
