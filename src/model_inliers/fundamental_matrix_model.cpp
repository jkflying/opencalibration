#include <opencalibration/model_inliers/fundamental_matrix_model.hpp>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <iostream>

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

fundamental_matrix_model::fundamental_matrix_model()
    : inlier_threshold(0.001), fundamental_matrix(Eigen::Matrix3d::Constant(NAN))
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

    for (size_t i = 0; i < num_inliers; i++)
    {
        Eigen::Vector2d p1 = corrs[i].measurement1.hnormalized();
        const double x = p1.x();
        const double y = p1.y();
        Eigen::Vector2d p2 = corrs[i].measurement2.hnormalized();
        const double x_ = p2.x();
        const double y_ = p2.y();

        A.row(i) << x * x_, x * y_, x, y * x_, y * y_, y, x_, y_, 1;
    }

    calculateFundamentalMatrix(A, fundamental_matrix);
}

size_t fundamental_matrix_model::evaluate(const std::vector<correspondence> &corrs, std::vector<bool> &inliers)
{
    inliers.resize(corrs.size());
    size_t count = 0;
    for (size_t i = 0; i < corrs.size(); i++)
    {
        bool in = error(corrs[i]) < inlier_threshold;
        inliers[i] = in;
        if (in)
            count++;
    }
    return count;
}

double fundamental_matrix_model::error(const correspondence &cor)
{
    return cor.measurement1.hnormalized().homogeneous().transpose() * fundamental_matrix *
           cor.measurement2.hnormalized().homogeneous();
}

} // namespace opencalibration
