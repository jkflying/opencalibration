#include <opencalibration/model_inliers/homography_model.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>

#include "model_utils.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace opencalibration
{

homography_model::homography_model()
    : homography(Eigen::Matrix3d::Constant(NAN)), homography_inverse(Eigen::Matrix3d::Constant(NAN))
{
}

namespace
{
// Least squares DLT with H(2,2) fixed to 1
template <typename Indices>
Eigen::Matrix3d solveHomography(const std::vector<correspondence> &corrs, const Indices &indices)
{
    Eigen::Matrix<double, Eigen::Dynamic, 9> P(indices.size() * 2, 9);
    Eigen::Index row = 0;
    for (size_t idx : indices)
    {
        const Eigen::Vector2d p1 = corrs[idx].measurement1.hnormalized();
        const Eigen::Vector2d p2 = corrs[idx].measurement2.hnormalized();
        const double x = p1.x(), y = p1.y(), x_ = p2.x(), y_ = p2.y();
        P.row(row++) << -x, -y, -1, 0, 0, 0, x * x_, y * x_, x_;
        P.row(row++) << 0, 0, 0, -x, -y, -1, x * y_, y * y_, y_;
    }

    Eigen::Matrix<double, 9, 1> h;
    h.head<8>() = P.leftCols<8>().colPivHouseholderQr().solve(-P.col(8));
    h(8) = 1;
    return Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(h.data());
}
} // namespace

void homography_model::fit(const std::vector<correspondence> &corrs,
                           const std::array<size_t, MINIMUM_POINTS> &initial_indices)
{
    homography = solveHomography(corrs, initial_indices);
    homography_inverse = homography.inverse();
}

void homography_model::fitInliers(const std::vector<correspondence> &corrs, const std::vector<bool> &inliers)
{
    const auto indices = inlierIndices(inliers);
    if (indices.size() < MINIMUM_POINTS)
        return;
    homography = solveHomography(corrs, indices);
    homography_inverse = homography.inverse();
}

double homography_model::error(const correspondence &corr)
{
    Eigen::Vector3d m1 = corr.measurement1 / corr.measurement1.z();
    Eigen::Vector3d m2 = corr.measurement2 / corr.measurement2.z();

    double fwd = ((homography * m1).hnormalized() - m2.head<2>()).squaredNorm();
    double bwd = ((homography_inverse * m2).hnormalized() - m1.head<2>()).squaredNorm();
    return std::sqrt((fwd + bwd) / 2.0);
}

double homography_model::evaluate(const std::vector<correspondence> &corrs, std::vector<bool> &inliers)
{
    return evaluateMsac(*this, corrs, inliers);
}

bool homography_model::checkSampleDegeneracy(const std::vector<correspondence> &corrs,
                                             const std::array<size_t, MINIMUM_POINTS> &indices)
{
    std::array<Eigen::Vector2d, 4> pts;
    for (size_t i = 0; i < 4; i++)
        pts[i] = corrs[indices[i]].measurement1.hnormalized();

    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++)
            for (int k = j + 1; k < 4; k++)
            {
                Eigen::Vector2d v1 = pts[j] - pts[i], v2 = pts[k] - pts[i];
                if (std::abs(v1.x() * v2.y() - v1.y() * v2.x()) < 1e-10)
                    return true; // degenerate
            }
    return false;
}

bool homography_model::decompose(const std::vector<correspondence> &corrs, const std::vector<bool> &inliers,
                                 std::array<decomposed_pose, 4> &poses)
{
    std::vector<cv::Mat> Rs_decomp, Ts_decomp, normals_decomp;
    cv::Mat h;
    cv::eigen2cv(homography, h);
    cv::Mat I;
    cv::eigen2cv(Eigen::Matrix3d::Identity().eval(), I);
    size_t solutions = cv::decomposeHomographyMat(h, I, Rs_decomp, Ts_decomp, normals_decomp);

    for (size_t i = 0; i < solutions; i++)
    {
        Eigen::Matrix3d R;
        cv::cv2eigen(Rs_decomp[i], R);

        Eigen::Vector3d T;
        cv::cv2eigen(Ts_decomp[i], T);

        Eigen::Vector3d N;
        cv::cv2eigen(normals_decomp[i], N);

        poses[i].score = 0;

        for (size_t j = 0; j < corrs.size(); j++)
        {
            if (!inliers[j])
            {
                continue;
            }
            double dot1 = N.dot(corrs[j].measurement1);
            double dot2 = (R * N).dot(corrs[j].measurement2);
            if (dot1 >= 0 && dot2 >= 0)
            {
                poses[i].score++;
            }
        }
        poses[i].orientation = Eigen::Quaterniond(R);
        poses[i].position = T;
    }
    for (size_t i = solutions; i < poses.size(); i++)
    {
        poses[i].score = -1;
    }
    std::stable_sort(poses.begin(), poses.end(),
                     [](const decomposed_pose &p1, const decomposed_pose &p2) { return p1.score >= p2.score; });

    return poses[0].score > 0;
}
} // namespace opencalibration
