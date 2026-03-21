#include <opencalibration/model_inliers/fundamental_matrix_model.hpp>
#include <opencalibration/model_inliers/homography_model.hpp>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <limits>

namespace
{
template <int D>
void calculateFundamentalMatrix(const Eigen::Matrix<double, D, 9> &A, Eigen::Matrix3d &fundamental_matrix)
{
    Eigen::Matrix<double, 9, 9> AtA = A.transpose() * A;

    Eigen::Matrix<double, 9, 1> F_ = AtA.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>().eval();
    fundamental_matrix.row(0) = F_.topRows<3>().transpose();
    fundamental_matrix.row(1) = F_.middleRows<3>(3).transpose();
    fundamental_matrix.row(2) = F_.bottomRows<3>().transpose();

    // Enforce rank-2 property of fundamental_matrix
    Eigen::JacobiSVD<Eigen::Matrix3d> constraints;
    constraints.compute(fundamental_matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Vector3d singularValues = constraints.singularValues();
    singularValues(2) = 0;
    fundamental_matrix = constraints.matrixU() * singularValues.asDiagonal() * constraints.matrixV().transpose();
}
} // namespace

namespace opencalibration
{

fundamental_matrix_model::fundamental_matrix_model() : fundamental_matrix(Eigen::Matrix3d::Constant(NAN))
{
}

void fundamental_matrix_model::fit(const std::vector<correspondence> &corrs,
                                   const std::array<size_t, MINIMUM_POINTS> &initial_indices)
{

    Eigen::Matrix<double, MINIMUM_POINTS, 9> A;

    for (size_t i = 0; i < MINIMUM_POINTS; i++)
    {

        Eigen::Vector2d p1 = corrs[initial_indices[i]].measurement1.hnormalized();
        const double x = p1.x();
        const double y = p1.y();
        Eigen::Vector2d p2 = corrs[initial_indices[i]].measurement2.hnormalized();
        const double x_ = p2.x();
        const double y_ = p2.y();

        A.row(i) << x * x_, x * y_, x, y * x_, y * y_, y, x_, y_, 1;
    }

    calculateFundamentalMatrix(A, fundamental_matrix);
}

void fundamental_matrix_model::fitInliers(const std::vector<correspondence> &corrs, const std::vector<bool> &inliers)
{
    size_t num_inliers = std::count(inliers.begin(), inliers.end(), true);

    if (num_inliers < MINIMUM_POINTS)
        return;

    Eigen::Matrix<double, Eigen::Dynamic, 9> A(num_inliers, 9);

    for (size_t i = 0, j = 0; i < corrs.size(); i++)
    {
        if (inliers[i])
        {
            Eigen::Vector2d p1 = corrs[i].measurement1.hnormalized();
            const double x = p1.x();
            const double y = p1.y();
            Eigen::Vector2d p2 = corrs[i].measurement2.hnormalized();
            const double x_ = p2.x();
            const double y_ = p2.y();

            A.row(j) << x * x_, x * y_, x, y * x_, y * y_, y, x_, y_, 1;
            j++;
        }
    }

    calculateFundamentalMatrix(A, fundamental_matrix);
}

double fundamental_matrix_model::evaluate(const std::vector<correspondence> &corrs, std::vector<bool> &inliers)
{
    inliers.resize(corrs.size());
    double total_score = 0;
    for (size_t i = 0; i < corrs.size(); i++)
    {
        double e = error(corrs[i]);
        if (e < inlier_threshold)
        {
            inliers[i] = true;
            double ratio = e / inlier_threshold;
            total_score += 1.0 - ratio * ratio;
        }
        else
        {
            inliers[i] = false;
        }
    }
    return total_score;
}

double fundamental_matrix_model::error(const correspondence &cor)
{
    Eigen::Vector3d x1 = cor.measurement1 / cor.measurement1.z();
    Eigen::Vector3d x2 = cor.measurement2 / cor.measurement2.z();
    double x2tFx1 = x2.transpose() * fundamental_matrix * x1;
    Eigen::Vector3d Fx1 = fundamental_matrix * x1;
    Eigen::Vector3d Ftx2 = fundamental_matrix.transpose() * x2;
    double denom = Fx1[0] * Fx1[0] + Fx1[1] * Fx1[1] + Ftx2[0] * Ftx2[0] + Ftx2[1] * Ftx2[1];
    if (denom < 1e-20)
        return std::numeric_limits<double>::max();
    return std::sqrt((x2tFx1 * x2tFx1) / denom);
}

void fundamental_matrix_model::checkDegeneracy(const std::vector<correspondence> &corrs, std::vector<bool> &inliers)
{
    // DEGENSAC: if F-inliers are dominated by a plane, recover F = [e']_x * H

    std::vector<size_t> f_inlier_idx;
    for (size_t i = 0; i < inliers.size(); i++)
    {
        if (inliers[i])
            f_inlier_idx.push_back(i);
    }

    if (f_inlier_idx.size() < homography_model::MINIMUM_POINTS)
        return;

    homography_model h_model;
    h_model.inlier_threshold = inlier_threshold * 2;

    std::array<size_t, 4> h_indices;
    for (size_t i = 0; i < 4; i++)
        h_indices[i] = f_inlier_idx[i];
    h_model.fit(corrs, h_indices);

    std::vector<bool> h_inliers(corrs.size(), false);
    size_t h_inlier_count = 0;
    for (size_t idx : f_inlier_idx)
    {
        if (h_model.error(corrs[idx]) < h_model.inlier_threshold)
        {
            h_inliers[idx] = true;
            h_inlier_count++;
        }
    }

    double h_ratio = static_cast<double>(h_inlier_count) / f_inlier_idx.size();
    if (h_ratio < 0.7)
        return;

    h_model.fitInliers(corrs, h_inliers);

    std::vector<size_t> non_h_idx;
    for (size_t idx : f_inlier_idx)
    {
        if (h_model.error(corrs[idx]) < h_model.inlier_threshold)
            h_inliers[idx] = true;
        else
        {
            h_inliers[idx] = false;
            non_h_idx.push_back(idx);
        }
    }

    if (non_h_idx.size() < 2)
        return;

    // Estimate epipole from off-plane points: (x2 x H*x1) . e' = 0
    Eigen::MatrixXd A(non_h_idx.size(), 3);
    for (size_t i = 0; i < non_h_idx.size(); i++)
    {
        const auto &m1 = corrs[non_h_idx[i]].measurement1;
        const auto &m2 = corrs[non_h_idx[i]].measurement2;
        Eigen::Vector3d x1 = m1 / m1.z();
        Eigen::Vector3d x2 = m2 / m2.z();
        A.row(i) = x2.cross(h_model.homography * x1).transpose();
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector3d epipole = svd.matrixV().rightCols<1>();

    // F = [e']_x * H, enforced to rank 2
    Eigen::Matrix3d e_cross;
    e_cross << 0, -epipole.z(), epipole.y(), epipole.z(), 0, -epipole.x(), -epipole.y(), epipole.x(), 0;
    Eigen::Matrix3d F_candidate = e_cross * h_model.homography;

    Eigen::JacobiSVD<Eigen::Matrix3d> f_svd(F_candidate, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singularValues = f_svd.singularValues();
    singularValues(2) = 0;
    F_candidate = f_svd.matrixU() * singularValues.asDiagonal() * f_svd.matrixV().transpose();

    // Keep the candidate only if it scores better
    Eigen::Matrix3d old_F = fundamental_matrix;
    std::vector<bool> old_inliers = inliers;

    fundamental_matrix = F_candidate;
    double candidate_score = evaluate(corrs, inliers);

    fundamental_matrix = old_F;
    double original_score = evaluate(corrs, old_inliers);

    if (candidate_score > original_score)
        fundamental_matrix = F_candidate;
    else
        inliers = old_inliers;
}

} // namespace opencalibration
