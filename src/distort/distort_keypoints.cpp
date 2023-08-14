#include <opencalibration/distort/distort_keypoints.hpp>

#include <ceres/tiny_solver.h>
#include <ceres/tiny_solver_autodiff_function.h>

#include <eigen3/Eigen/Geometry>

namespace
{
using namespace opencalibration;

struct DistortionFunctor
{
    enum
    {
        NUM_PARAMETERS = 2,
        NUM_RESIDUALS = 2
    };

    DistortionFunctor(const Eigen::Matrix<double, NUM_PARAMETERS, 1> &projected_point,
                      const Eigen::Vector3d &radial_distortion, const Eigen::Vector2d &tangential_distortion)
        : projected_point(projected_point), radial_distortion(radial_distortion),
          tangential_distortion(tangential_distortion)
    {
    }

    template <typename Scalar> bool operator()(const Scalar *parameters, Scalar *residuals) const
    {
        Eigen::Map<const Eigen::Matrix<Scalar, NUM_PARAMETERS, 1>> param_map(parameters);
        Eigen::Map<Eigen::Matrix<Scalar, NUM_RESIDUALS, 1>> res_map(residuals);

        res_map = projected_point - distortProjectedRay<Scalar>(param_map, radial_distortion.cast<Scalar>(),
                                                                tangential_distortion.cast<Scalar>());

        return true;
    }

  private:
    const Eigen::Matrix<double, NUM_PARAMETERS, 1> &projected_point;
    const Eigen::Vector3d &radial_distortion;
    const Eigen::Vector2d &tangential_distortion;
};

} // namespace

namespace opencalibration
{
std::vector<correspondence> distort_keypoints(const std::vector<feature_2d> &features1,
                                              const std::vector<feature_2d> &features2,
                                              const std::vector<feature_match> &matches,
                                              const DifferentiableCameraModel<double> &model1,
                                              const DifferentiableCameraModel<double> &model2)
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

Eigen::Vector3d image_to_3d(const Eigen::Vector2d &keypoint, const DifferentiableCameraModel<double> &model)
{

    Eigen::Vector2d unprojected_point = (keypoint - model.principle_point) / model.focal_length_pixels;

    Eigen::Vector2d undistorted_point = unprojected_point;

    // TODO: make better initial guesses depending on the structure of the distortion parameters
    // eg. use a quadratic solver if only 2/4 or 1/2 nonzero, and benchmark to see if it is faster

    if ((model.radial_distortion.array() != 0).any() || (model.tangential_distortion.array() != 0).any())
    {
        using ADFunctor = ceres::TinySolverAutoDiffFunction<DistortionFunctor, 2, 2>;

        DistortionFunctor func(unprojected_point, model.radial_distortion, model.tangential_distortion);
        ADFunctor adfunc(func);

        ceres::TinySolver<ADFunctor> solver;
        solver.options.parameter_tolerance = 1e-2 / (model.principle_point.norm() + model.focal_length_pixels);
        solver.options.gradient_tolerance = solver.options.parameter_tolerance * 1e-2;
        solver.options.max_num_iterations = 10;
        solver.options.cost_threshold = 1e-16;
        solver.Solve(adfunc, &undistorted_point);
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

Eigen::Vector2d image_from_3d(const Eigen::Vector3d &ray, const InverseDifferentiableCameraModel<double> &model)
{
    Eigen::Vector2d ray_projected;
    switch (model.projection_type)
    {
    case ProjectionType::PLANAR: {
        ray_projected = ray.hnormalized();
        break;
    }
    case ProjectionType::UNKNOWN:
        ray_projected.fill(NAN);
        break;
    }

    Eigen::Vector2d ray_distorted = ray_projected;

    if ((model.radial_distortion.array() != 0).any() || (model.tangential_distortion.array() != 0).any())
    {
        using ADFunctor = ceres::TinySolverAutoDiffFunction<DistortionFunctor, 2, 2>;

        DistortionFunctor func(ray_projected, model.radial_distortion, model.tangential_distortion);
        ADFunctor adfunc(func);
        ceres::TinySolver<ADFunctor> solver;

        solver.options.parameter_tolerance = 1e-2 / (model.principle_point.norm() + model.focal_length_pixels);
        solver.options.gradient_tolerance = solver.options.parameter_tolerance * 1e-2;
        solver.options.max_num_iterations = 10;
        solver.options.cost_threshold = 1e-16;
        solver.Solve(adfunc, &ray_distorted);
    }

    const Eigen::Vector2d pixel_location = ray_distorted * model.focal_length_pixels + model.principle_point;
    return pixel_location;
}
} // namespace opencalibration
