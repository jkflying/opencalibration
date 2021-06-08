#include <opencalibration/geometry/intersection.hpp>

#include <Eigen/Geometry>

namespace opencalibration
{
std::pair<Eigen::Vector3d, double> rayIntersection(const ray_d &r1, const ray_d &r2)
{
    Eigen::Vector3d res{NAN, NAN, NAN};
    double error = NAN;

    double n1dn1 = r1.dir.dot(r1.dir);
    double n1dn2 = r1.dir.dot(r2.dir);
    double n2dn2 = r2.dir.dot(r2.dir);

    auto sqr = [](double d) { return d * d; };
    double scale_denom = sqr(n1dn2) - n1dn1 * n2dn2;

    if (scale_denom < 0.01) // signal to noise ratio is too low
    {

        Eigen::Vector3d offset = (r2.offset - r1.offset);
        double offset1 = offset.dot(r1.dir);
        double offset2 = offset.dot(r2.dir);
        double norm_scale = 1. / scale_denom;

        double t = (offset2 * n1dn2 - offset1 * n2dn2) * norm_scale;
        double s = (offset2 * n1dn1 - offset1 * n1dn2) * norm_scale;

        Eigen::Vector3d p1 = (r1.offset + t * r1.dir), p2 = (r2.offset + s * r2.dir);
        res = 0.5 * (p1 + p2);
        error = (p1 - p2).squaredNorm() * (t >= 0 && s >= 0 ? 1 : -1);
    }

    return std::make_pair(res, error);
}

std::pair<Eigen::Vector3d, double> rayIntersection(const std::vector<ray_d> &rays)
{
    Eigen::Vector3d res{NAN, NAN, NAN};
    double error = NAN;

    if (rays.size() > 1)
    {
        auto init = rayIntersection(rays[0], rays[1]);
        res = init.first;
        error = init.second;

        // TODO: better optimization taking into account multiple rays and a robust cost function
    }

    return std::make_pair(res, error);
}

} // namespace opencalibration
