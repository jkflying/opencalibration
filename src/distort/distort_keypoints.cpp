#include <opencalibration/distort/distort_keypoints.hpp>

#include <ceres/jet.h>
#include <ceres/tiny_solver.h>
#include <eigen3/Eigen/Geometry>

namespace
{
using namespace opencalibration;

struct DistortionFunctor
{
    using Scalar = double;
    enum
    {
        NUM_PARAMETERS = 2,
        NUM_RESIDUALS = 2
    };

    DistortionFunctor(const Eigen::Matrix<Scalar, NUM_PARAMETERS, 1> &projected_point,
                      const DifferentiableCameraModel<Scalar> &model)
        : projected_point(projected_point), model(model)
    {
    }

    bool operator()(const Scalar *parameters, Scalar *residuals, Scalar *jacobian) const
    {
        Eigen::Map<const Eigen::Matrix<Scalar, NUM_PARAMETERS, 1>> param_map(parameters);
        Eigen::Map<Eigen::Matrix<Scalar, NUM_RESIDUALS, 1>> res_map(residuals);
        if (jacobian == nullptr)
        {
            res_map = projected_point - distortProjectedRay<Scalar>(param_map, model);
        }
        else
        {
            using T = ceres::Jet<Scalar, NUM_PARAMETERS>;
            Eigen::Matrix<T, NUM_PARAMETERS, 1> param_t = param_map.cast<T>();
            for (Eigen::Index i = 0; i < NUM_PARAMETERS; i++)
            {
                param_t[i].v[i] = 1;
            }

            Eigen::Matrix<T, NUM_RESIDUALS, 1> res_t =
                projected_point.cast<T>() - distortProjectedRay<T>(param_t, model.cast<T>());

            Eigen::Map<Eigen::Matrix<Scalar, NUM_PARAMETERS, NUM_RESIDUALS>> jac_map(jacobian);
            for (Eigen::Index i = 0; i < NUM_RESIDUALS; i++)
            {
                res_map[i] = res_t[i].a;
                jac_map.row(i) = res_t[i].v;
            }
        }
        return true;
    }

  private:
    const Eigen::Matrix<Scalar, NUM_PARAMETERS, 1> &projected_point;
    const DifferentiableCameraModel<Scalar> &model;
};

} // namespace

namespace opencalibration
{
std::vector<correspondence> distort_keypoints(const std::vector<feature_2d> &features1,
                                              const std::vector<feature_2d> &features2,
                                              const std::vector<feature_match> &matches, const CameraModel &model1,
                                              const CameraModel &model2)
{
    std::vector<correspondence> distorted;
    distorted.reserve(matches.size());
    for (const feature_match &m : matches)
    {
        correspondence cor;
        cor.measurement1 = image_to_3d(features1[m.feature_index_1].location, model1);
        cor.measurement2 = image_to_3d(features2[m.feature_index_2].location, model2);
        distorted.push_back(cor);
    }

    return distorted;
}

Eigen::Vector3d image_to_3d(const Eigen::Vector2d &keypoint, const CameraModel &model)
{

    Eigen::Vector2d unprojected_point = (keypoint - model.principle_point) / model.focal_length_pixels;

    Eigen::Vector2d undistorted_point = unprojected_point;

    // TODO: make better initial guesses depending on the structure of the distortion parameters
    // eg. use a quadratic solver if only 2/4 or 1/2 nonzero, and benchmark to see if it is faster

    if (model.radial_distortion.squaredNorm() + model.tangential_distortion.squaredNorm() > 0)
    {
        ceres::TinySolver<DistortionFunctor> solver;
        DistortionFunctor func(unprojected_point, model);
        solver.options.parameter_tolerance = 1e-2 / (model.principle_point.norm() + model.focal_length_pixels);
        solver.options.gradient_tolerance = solver.options.parameter_tolerance * 1e-2;
        solver.options.max_num_iterations = 10;
        solver.options.cost_threshold = 1e-16;
        solver.Solve(func, &undistorted_point);
    }

    Eigen::Vector3d ray;
    switch (model.projection_type)
    {
    case ProjectionType::PLANAR:
        ray = undistorted_point.homogeneous().normalized();
        break;
    case ProjectionType::UNKNOWN:
        break;
    }
    return ray;
}
} // namespace opencalibration
