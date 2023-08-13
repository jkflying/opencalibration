#include <opencalibration/distort/invert_distortion.hpp>

#include <opencalibration/distort/distort_keypoints.hpp>

#include <ceres/jet.h>
#include <ceres/tiny_solver.h>

namespace
{

struct InverseDistortionFunctor
{
    using Scalar = double;
    enum
    {
        NUM_PARAMETERS = 5,
        NUM_RESIDUALS = Eigen::Dynamic
    };

    bool operator()(const Scalar *parameters, Scalar *residuals, Scalar *jacobian) const
    {

        Eigen::Map<const Eigen::Matrix<Scalar, NUM_PARAMETERS, 1>> param_map(parameters);
        Eigen::Map<Eigen::VectorXd> res_map(residuals, NumResiduals());

        if (jacobian == nullptr)
        {
            opencalibration::InverseDifferentiableCameraModel<double> invertedWithDistortion = invertedNoDistortion;
            invertedWithDistortion.radial_distortion = param_map.topRows<3>();
            invertedWithDistortion.tangential_distortion = param_map.bottomRows<2>();
            for (size_t i = 0; i < correspondences.size(); i++)
            {
                const Eigen::Vector3d error_3d =
                    image_to_3d(correspondences[i].second, invertedWithDistortion) - correspondences[i].first;
                res_map.block<3, 1>(i * 3, 0) = error_3d;
            }
        }
        else
        {
            using T = ceres::Jet<Scalar, NUM_PARAMETERS>;
            opencalibration::InverseDifferentiableCameraModel<T> invertedWithDistortion =
                invertedNoDistortion.cast<T>();
            invertedWithDistortion.radial_distortion = param_map.topRows<3>().cast<T>();
            invertedWithDistortion.tangential_distortion = param_map.bottomRows<2>().cast<T>();
            for (int i = 0; i < 3; i++)
                invertedWithDistortion.radial_distortion(i).v(i) = 1;
            for (int i = 1; i < 2; i++)
                invertedWithDistortion.tangential_distortion(i).v(i + 3) = 1;

            for (size_t i = 0; i < correspondences.size(); i++)
            {
                const Eigen::Matrix<T, 2, 1> source = correspondences[i].second.cast<T>();
                const Eigen::Matrix<T, 3, 1> value_3d = image_to_3d(source, invertedWithDistortion);
                const Eigen::Matrix<T, 3, 1> error_3d = value_3d - correspondences[i].first.cast<T>();

                Eigen::Map<Eigen::Matrix<Scalar, NUM_RESIDUALS, NUM_PARAMETERS>> jac_map(jacobian, NumResiduals(),
                                                                                         NUM_PARAMETERS);
                for (size_t j = 0; i < 3; i++)
                {
                    res_map(i * 3 + j, 0) = error_3d(j).a;
                    jac_map.row(i * 3 + j) = error_3d(j).v;
                }
            }
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
    inverted.radial_distortion = {};
    inverted.tangential_distortion = {};

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

    ceres::TinySolver<InverseDistortionFunctor> solver;
    InverseDistortionFunctor functor(correspondences, inverted);

    Eigen::Matrix<double, 5, 1> parameters;
    parameters.fill(0);
    solver.Solve(functor, &parameters);

    inverted.radial_distortion = parameters.topRows<3>();
    inverted.tangential_distortion = parameters.bottomRows<2>();

    return inverted;
}

DifferentiableCameraModel<double> convertModel(const InverseDifferentiableCameraModel<double> &invertedModel)
{
    DifferentiableCameraModel<double> standard = invertedModel.cast<double, CameraModelTag::FORWARD>();

    standard.radial_distortion = {};
    standard.tangential_distortion = {};
    // TODO: invert the distortion variables so that the model can be used in standard projections differentiably

    return standard;
}

} // namespace opencalibration
