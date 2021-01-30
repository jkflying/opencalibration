#include <opencalibration/geometry/intersection.hpp>

#include <Eigen/Geometry>

namespace opencalibration
{
Eigen::Vector4d rayIntersection(const Eigen::Vector3d &origin1, const Eigen::Vector3d &normal1,
                                const Eigen::Vector3d &origin2, const Eigen::Vector3d &normal2)
{
    Eigen::Vector4d res{NAN, NAN, NAN, NAN};

    auto sqr = [](double d) { return d * d; };
    double scale_denom = sqr(normal1.dot(normal2)) - normal1.dot(normal1) * normal2.dot(normal2);

    if (scale_denom < 0.01)
    {

        Eigen::Vector3d offset = (origin2 - origin1);
        double offset1 = offset.dot(normal1);
        double offset2 = offset.dot(normal2);
        double norm_scale = 1. / scale_denom;

        double t = (offset2 * normal1.dot(normal2) - offset1 * normal2.dot(normal2)) * norm_scale;
        double s = (offset2 * normal1.dot(normal1) - offset1 * normal1.dot(normal2)) * norm_scale;

        Eigen::Vector3d p1 = (origin1 + t * normal1), p2 = (origin2 + s * normal2);
        res.topRows<3>() = 0.5 * (p1 + p2);
        res(3) = (p1 - p2).norm();
    }

    return res;
}

} // namespace opencalibration
