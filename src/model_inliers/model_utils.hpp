#pragma once

#include <opencalibration/types/correspondence.hpp>

#include <eigen3/Eigen/Dense>

#include <limits>
#include <vector>

namespace opencalibration
{

inline double msacScore(double error, double threshold)
{
    const double ratio = error / threshold;
    return 1.0 - ratio * ratio;
}

template <typename Model>
double evaluateMsac(Model &model, const std::vector<correspondence> &corrs, std::vector<bool> &inliers)
{
    inliers.resize(corrs.size());
    double total_score = 0;
    for (size_t i = 0; i < corrs.size(); i++)
    {
        const double e = model.error(corrs[i]);
        inliers[i] = e < model.inlier_threshold;
        if (inliers[i])
            total_score += msacScore(e, model.inlier_threshold);
    }
    return total_score;
}

inline std::vector<size_t> inlierIndices(const std::vector<bool> &inliers)
{
    std::vector<size_t> indices;
    for (size_t i = 0; i < inliers.size(); i++)
        if (inliers[i])
            indices.push_back(i);
    return indices;
}

// Least-squares M (up to scale) with x2^T M x1 = 0, before any rank constraint
template <typename Indices>
Eigen::Matrix3d solveEpipolarConstraint(const std::vector<correspondence> &corrs, const Indices &indices)
{
    Eigen::Matrix<double, Eigen::Dynamic, 9> A(indices.size(), 9);
    Eigen::Index row = 0;
    for (size_t idx : indices)
    {
        const Eigen::Vector3d x1 = corrs[idx].measurement1 / corrs[idx].measurement1.z();
        const Eigen::Vector3d x2 = corrs[idx].measurement2 / corrs[idx].measurement2.z();
        A.row(row++) << x2.x() * x1.transpose(), x2.y() * x1.transpose(), x1.transpose();
    }

    const Eigen::Matrix<double, 9, 9> AtA = A.transpose() * A;
    const Eigen::Matrix<double, 9, 1> m = AtA.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    return Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(m.data());
}

inline double sampsonError(const Eigen::Matrix3d &M, const correspondence &cor)
{
    const Eigen::Vector3d x1 = cor.measurement1 / cor.measurement1.z();
    const Eigen::Vector3d x2 = cor.measurement2 / cor.measurement2.z();
    const double x2tMx1 = x2.dot(M * x1);
    const Eigen::Vector3d Mx1 = M * x1;
    const Eigen::Vector3d Mtx2 = M.transpose() * x2;
    const double denom = Mx1.head<2>().squaredNorm() + Mtx2.head<2>().squaredNorm();
    if (denom < 1e-20)
        return std::numeric_limits<double>::max();
    return std::sqrt(x2tMx1 * x2tMx1 / denom);
}

template <typename Adjust> Eigen::Matrix3d adjustSingularValues(const Eigen::Matrix3d &M, Adjust adjust)
{
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = svd.singularValues();
    adjust(singular_values);
    return svd.matrixU() * singular_values.asDiagonal() * svd.matrixV().transpose();
}

inline Eigen::Matrix3d enforceRankTwo(const Eigen::Matrix3d &M)
{
    return adjustSingularValues(M, [](Eigen::Vector3d &s) { s(2) = 0; });
}

inline Eigen::Matrix3d crossProductMatrix(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}

} // namespace opencalibration
