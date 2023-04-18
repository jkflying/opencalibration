#include <opencalibration/model_inliers/homography_model.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace opencalibration
{

homography_model::homography_model() : inlier_threshold(0.005), homography(Eigen::Matrix3d::Constant(NAN))
{
}

void homography_model::fit(const std::vector<correspondence> &corrs,
                           const std::array<size_t, MINIMUM_POINTS> &initial_indices)
{
    Eigen::Matrix<double, 9, 9> P;

    for (size_t i = 0; i < 4; i++)
    {
        Eigen::Vector2d p1 = corrs[initial_indices[i]].measurement1.hnormalized();
        const double x = p1.x();
        const double y = p1.y();
        Eigen::Vector2d p2 = corrs[initial_indices[i]].measurement2.hnormalized();
        const double x_ = p2.x();
        const double y_ = p2.y();

        P.row(i * 2) << -x, -y, -1, 0, 0, 0, x * x_, y * x_, x_;
        P.row(i * 2 + 1) << 0, 0, 0, -x, -y, -1, x * y_, y * y_, y_;
    }

    // add constraint that bottom right corner is 1
    P.bottomRows<1>().setZero();
    P.bottomRightCorner<1, 1>() << 1;
    Eigen::Matrix<double, 9, 1> rhs;
    rhs.setZero();
    rhs.bottomRows<1>() << 1;

    Eigen::Matrix<double, 9, 1> H_ = P.fullPivLu().solve(rhs);
    homography.row(0) = H_.topRows<3>().transpose();
    homography.row(1) = H_.middleRows<3>(3).transpose();
    homography.row(2) = H_.bottomRows<3>().transpose();
    homography /= homography(2, 2); // renormalize in case that constraint wasn't enough
}

void homography_model::fitInliers(const std::vector<correspondence> &corrs, const std::vector<bool> &inliers)
{
    size_t num_inliers = std::count(inliers.begin(), inliers.end(), true);
    Eigen::Matrix<double, Eigen::Dynamic, 9> P(num_inliers * 2 + 1, 9);

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

            P.row(j * 2) << -x, -y, -1, 0, 0, 0, x * x_, y * x_, x_;
            P.row(j * 2 + 1) << 0, 0, 0, -x, -y, -1, x * y_, y * y_, y_;

            j++;
        }
    }

    // add constraint that bottom right corner is 1
    P.bottomRows<1>() << 0, 0, 0, 0, 0, 0, 0, 0, 1;
    Eigen::Matrix<double, Eigen::Dynamic, 1> rhs(num_inliers * 2 + 1, 1);
    rhs.setZero();
    rhs.bottomRows<1>() << 1;

    Eigen::Matrix<double, 9, 1> H_ = P.fullPivLu().solve(rhs);
    homography.row(0) = H_.topRows<3>().transpose();
    homography.row(1) = H_.middleRows<3>(3).transpose();
    homography.row(2) = H_.bottomRows<3>().transpose();
    homography /= homography(2, 2); // renormalize in case that constraint wasn't enough
}

double homography_model::error(const correspondence &corr)
{
    return ((homography * corr.measurement1.hnormalized().homogeneous()).hnormalized() -
            corr.measurement2.hnormalized())
        .norm();
}
size_t homography_model::evaluate(const std::vector<correspondence> &corrs, std::vector<bool> &inliers)
{
    size_t num_inliers = 0;
    for (size_t i = 0; i < inliers.size(); i++)
    {
        bool in = error(corrs[i]) < inlier_threshold;
        inliers[i] = in;
        num_inliers += in;
    }
    return num_inliers;
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
