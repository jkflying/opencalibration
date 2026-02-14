#include <opencalibration/model_inliers/essential_matrix_model.hpp>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <cmath>
#include <iostream>

namespace
{
template <int D> void calculateEssentialMatrix(const Eigen::Matrix<double, D, 9> &A, Eigen::Matrix3d &essential_matrix)
{
    Eigen::Matrix<double, 9, 9> AtA = A.transpose() * A;

    Eigen::Matrix<double, 9, 1> E_ = AtA.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>().eval();
    essential_matrix.row(0) = E_.topRows<3>().transpose();
    essential_matrix.row(1) = E_.middleRows<3>(3).transpose();
    essential_matrix.row(2) = E_.bottomRows<3>().transpose();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd;
    svd.compute(essential_matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Vector3d singularValues = svd.singularValues();

    double avg = (singularValues(0) + singularValues(1)) / 2.0;
    singularValues(0) = avg;
    singularValues(1) = avg;
    singularValues(2) = 0;

    essential_matrix = svd.matrixU() * singularValues.asDiagonal() * svd.matrixV().transpose();
}
} // namespace

namespace opencalibration
{

essential_matrix_model::essential_matrix_model()
    : inlier_threshold(0.001), essential_matrix(Eigen::Matrix3d::Constant(NAN))
{
}

void essential_matrix_model::fit(const std::vector<correspondence> &corrs,
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

    calculateEssentialMatrix(A, essential_matrix);
}

void essential_matrix_model::fitInliers(const std::vector<correspondence> &corrs, const std::vector<bool> &inliers)
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

    calculateEssentialMatrix(A, essential_matrix);
}

size_t essential_matrix_model::evaluate(const std::vector<correspondence> &corrs, std::vector<bool> &inliers)
{
    inliers.resize(corrs.size());
    size_t count = 0;
    for (size_t i = 0; i < corrs.size(); i++)
    {
        bool in = std::abs(error(corrs[i])) < inlier_threshold;
        inliers[i] = in;
        if (in)
            count++;
    }
    return count;
}

double essential_matrix_model::error(const correspondence &cor)
{
    return cor.measurement1.hnormalized().homogeneous().transpose() * essential_matrix *
           cor.measurement2.hnormalized().homogeneous();
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
