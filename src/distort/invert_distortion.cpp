#include <opencalibration/distort/invert_distortion.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>

#include <ceres/tiny_solver.h>
#include <ceres/tiny_solver_autodiff_function.h>

namespace
{

struct InverseDistortionFunctor
{
    enum
    {
        NUM_PARAMETERS = 5,
        NUM_RESIDUALS = Eigen::Dynamic
    };

    template <typename Scalar> bool operator()(const Scalar *parameters, Scalar *residuals) const
    {

        Eigen::Map<const Eigen::Matrix<Scalar, NUM_PARAMETERS, 1>> param_map(parameters);
        Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> res_map(residuals, NumResiduals());

        opencalibration::InverseDifferentiableCameraModel<Scalar> invertedWithDistortion =
            invertedNoDistortion.cast<Scalar>();
        invertedWithDistortion.radial_distortion = param_map.template topRows<3>();
        invertedWithDistortion.tangential_distortion = param_map.template bottomRows<2>();
        for (size_t i = 0; i < correspondences.size(); i++)
        {
            const Eigen::Matrix<Scalar, 3, 1> error_3d =
                opencalibration::image_to_3d<Scalar>(correspondences[i].second.cast<Scalar>(), invertedWithDistortion) -
                correspondences[i].first.cast<Scalar>();
            res_map.template block<3, 1>(i * 3, 0) = error_3d;
        }

        return true;
    }

    int NumResiduals() const
    {
        return static_cast<int>(correspondences.size()) * 3;
    }

    InverseDistortionFunctor(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector2d>> &correspondences,
                             const opencalibration::InverseDifferentiableCameraModel<double> &noDistortionModel)
        : correspondences(correspondences), invertedNoDistortion(noDistortionModel)
    {
    }

  private:
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector2d>> &correspondences;
    const opencalibration::InverseDifferentiableCameraModel<double> &invertedNoDistortion;
};

} // namespace

namespace opencalibration
{

InverseDifferentiableCameraModel<double> convertModel(const DifferentiableCameraModel<double> &standardModel)
{
    InverseDifferentiableCameraModel<double> inverted = standardModel.cast<double, CameraModelTag::INVERSE>();
    inverted.radial_distortion.fill(0);
    inverted.tangential_distortion.fill(0);

    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector2d>> correspondences;

    static constexpr int grid_divisions = 20;

    correspondences.reserve(standardModel.pixels_cols / grid_divisions * standardModel.pixels_rows / grid_divisions);

    for (size_t i = 0; i < standardModel.pixels_cols; i += standardModel.pixels_cols / grid_divisions)
    {
        for (size_t j = 0; j < standardModel.pixels_rows; j += standardModel.pixels_rows / grid_divisions)
        {
            const Eigen::Vector2d p(i, j);
            const Eigen::Vector3d training_point =
                image_to_3d(p, standardModel); // TODO iterate over the 3d point directly instead of having to invert
                                               // the forward model to find it imprecisely

            // The `training_point` is only approximated, so we need to get the exact value of the forward model from
            // that point
            const Eigen::Vector2d p2 = image_from_3d(training_point, standardModel);

            if (!training_point.hasNaN())
            {
                correspondences.emplace_back(training_point, p2);
            }
        }
    }

    using ADFunctor = ceres::TinySolverAutoDiffFunction<InverseDistortionFunctor, Eigen::Dynamic, 5>;
    ceres::TinySolver<ADFunctor> solver;
    InverseDistortionFunctor functor(correspondences, inverted);
    ADFunctor autoDiffFunctor(functor);

    Eigen::Matrix<double, 5, 1> parameters;
    parameters.fill(0);
    solver.Solve(autoDiffFunctor, &parameters);

    inverted.radial_distortion = parameters.topRows<3>();
    inverted.tangential_distortion = parameters.bottomRows<2>();

    return inverted;
}

DifferentiableCameraModel<double> convertModel(const InverseDifferentiableCameraModel<double> &invertedModel)
{
    DifferentiableCameraModel<double> standard = invertedModel.cast<double, CameraModelTag::FORWARD>();

    standard.radial_distortion.fill(0);
    standard.tangential_distortion.fill(0);
    // TODO: invert the distortion variables so that the model can be used in standard projections differentiably

    return standard;
}

} // namespace opencalibration
