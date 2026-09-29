#include <opencalibration/model_inliers/fundamental_matrix_model.hpp>
#include <opencalibration/model_inliers/ransac.hpp>

#include "model_utils.hpp"

namespace opencalibration
{

fundamental_matrix_model::fundamental_matrix_model() : fundamental_matrix(Eigen::Matrix3d::Constant(NAN))
{
}

void fundamental_matrix_model::fit(const std::vector<correspondence> &corrs,
                                   const std::array<size_t, MINIMUM_POINTS> &initial_indices)
{
    fundamental_matrix = enforceRankTwo(solveEpipolarConstraint(corrs, initial_indices));
}

void fundamental_matrix_model::fitInliers(const std::vector<correspondence> &corrs, const std::vector<bool> &inliers)
{
    const auto indices = inlierIndices(inliers);
    if (indices.size() < MINIMUM_POINTS)
        return;
    fundamental_matrix = enforceRankTwo(solveEpipolarConstraint(corrs, indices));
}

double fundamental_matrix_model::evaluate(const std::vector<correspondence> &corrs, std::vector<bool> &inliers)
{
    return evaluateMsac(*this, corrs, inliers);
}

double fundamental_matrix_model::error(const correspondence &cor)
{
    return sampsonError(fundamental_matrix, cor);
}

void fundamental_matrix_model::checkDegeneracy(const std::vector<correspondence> &corrs, std::vector<bool> &inliers)
{
    // DEGENSAC: if F-inliers are dominated by a plane, recover F = [e']_x * H
    const std::vector<size_t> f_inlier_idx = inlierIndices(inliers);
    if (f_inlier_idx.size() < homography_model::MINIMUM_POINTS)
        return;

    std::vector<correspondence> f_inlier_corrs;
    f_inlier_corrs.reserve(f_inlier_idx.size());
    for (size_t idx : f_inlier_idx)
        f_inlier_corrs.push_back(corrs[idx]);

    homography_model h_model;
    h_model.inlier_threshold = inlier_threshold * 2;
    std::vector<bool> on_plane;
    ransac(f_inlier_corrs, h_model, on_plane);

    const auto plane_count = static_cast<size_t>(std::count(on_plane.begin(), on_plane.end(), true));
    const size_t off_plane_count = f_inlier_corrs.size() - plane_count;
    if (plane_count < 0.7 * f_inlier_corrs.size() || off_plane_count < 2)
        return;

    // Epipole from off-plane points: (x2 x H*x1) . e' = 0
    Eigen::MatrixXd A(off_plane_count, 3);
    for (size_t i = 0, row = 0; i < f_inlier_corrs.size(); i++)
    {
        if (on_plane[i])
            continue;
        const Eigen::Vector3d x1 = f_inlier_corrs[i].measurement1 / f_inlier_corrs[i].measurement1.z();
        const Eigen::Vector3d x2 = f_inlier_corrs[i].measurement2 / f_inlier_corrs[i].measurement2.z();
        A.row(row++) = x2.cross(h_model.homography * x1).transpose();
    }
    const Eigen::Vector3d epipole = Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeFullV).matrixV().rightCols<1>();

    const Eigen::Matrix3d old_F = fundamental_matrix;
    std::vector<bool> old_inliers = inliers;
    const double original_score = evaluate(corrs, old_inliers);

    fundamental_matrix = enforceRankTwo(crossProductMatrix(epipole) * h_model.homography);
    const double candidate_score = evaluate(corrs, inliers);

    if (candidate_score <= original_score)
    {
        fundamental_matrix = old_F;
        inliers = old_inliers;
    }
}

} // namespace opencalibration
