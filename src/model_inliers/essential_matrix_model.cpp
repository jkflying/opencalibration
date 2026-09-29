#include <opencalibration/model_inliers/essential_matrix_model.hpp>

#include "model_utils.hpp"

namespace opencalibration
{
namespace
{
Eigen::Matrix3d toEssential(const Eigen::Matrix3d &M)
{
    return adjustSingularValues(M, [](Eigen::Vector3d &s) {
        const double avg = (s(0) + s(1)) / 2.0;
        s << avg, avg, 0;
    });
}
} // namespace

essential_matrix_model::essential_matrix_model() : essential_matrix(Eigen::Matrix3d::Constant(NAN))
{
}

void essential_matrix_model::fit(const std::vector<correspondence> &corrs,
                                 const std::array<size_t, MINIMUM_POINTS> &initial_indices)
{
    essential_matrix = toEssential(solveEpipolarConstraint(corrs, initial_indices));
}

void essential_matrix_model::fitInliers(const std::vector<correspondence> &corrs, const std::vector<bool> &inliers)
{
    const auto indices = inlierIndices(inliers);
    if (indices.size() < MINIMUM_POINTS)
        return;
    essential_matrix = toEssential(solveEpipolarConstraint(corrs, indices));
}

double essential_matrix_model::evaluate(const std::vector<correspondence> &corrs, std::vector<bool> &inliers)
{
    return evaluateMsac(*this, corrs, inliers);
}

double essential_matrix_model::error(const correspondence &cor)
{
    return sampsonError(essential_matrix, cor);
}

bool essential_matrix_model::decompose(const std::vector<correspondence> & /*corrs*/,
                                       const std::vector<bool> & /*inliers*/, std::array<decomposed_pose, 4> &poses)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(essential_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3d W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    Eigen::Matrix3d R1 = svd.matrixU() * W * svd.matrixV().transpose();
    Eigen::Matrix3d R2 = svd.matrixU() * W.transpose() * svd.matrixV().transpose();

    if (R1.determinant() < 0)
        R1 = -R1;
    if (R2.determinant() < 0)
        R2 = -R2;

    Eigen::Vector3d t = svd.matrixU().col(2);

    poses[0].orientation = Eigen::Quaterniond(R1);
    poses[0].position = t;
    poses[1].orientation = Eigen::Quaterniond(R1);
    poses[1].position = -t;
    poses[2].orientation = Eigen::Quaterniond(R2);
    poses[2].position = t;
    poses[3].orientation = Eigen::Quaterniond(R2);
    poses[3].position = -t;

    return true;
}

} // namespace opencalibration
