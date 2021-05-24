#include <opencalibration/geometry/intersection.hpp>

#include <Eigen/Geometry>

namespace opencalibration
{
std::pair<Eigen::Vector3d, double> rayIntersection(const Eigen::Vector3d &origin1, const Eigen::Vector3d &normal1,
                                                   const Eigen::Vector3d &origin2, const Eigen::Vector3d &normal2)
{
    Eigen::Vector3d res{NAN, NAN, NAN};
    double error = NAN;

    double n1dn1 = normal1.dot(normal1);
    double n1dn2 = normal1.dot(normal2);
    double n2dn2 = normal2.dot(normal2);

    auto sqr = [](double d) { return d * d; };
    double scale_denom = sqr(n1dn2) - n1dn1 * n2dn2;

    if (scale_denom < 0.01) // signal to noise ratio is too low
    {

        Eigen::Vector3d offset = (origin2 - origin1);
        double offset1 = offset.dot(normal1);
        double offset2 = offset.dot(normal2);
        double norm_scale = 1. / scale_denom;

        double t = (offset2 * n1dn2 - offset1 * n2dn2) * norm_scale;
        double s = (offset2 * n1dn1 - offset1 * n1dn2) * norm_scale;

        Eigen::Vector3d p1 = (origin1 + t * normal1), p2 = (origin2 + s * normal2);
        res.topRows<3>() = 0.5 * (p1 + p2);
        error = (p1 - p2).squaredNorm() * (t >= 0 && s >= 0 ? 1 : -1);
    }

    return std::make_pair(res, error);
}

} // namespace opencalibration
