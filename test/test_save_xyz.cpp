#include <opencalibration/io/saveXYZ.hpp>

#include <gtest/gtest.h>

#include <sstream>

using namespace opencalibration;

namespace
{
std::vector<surface_model> surfacesWith(const point_cloud &cloud)
{
    surface_model surface;
    surface.cloud.push_back(cloud);
    return {surface};
}

size_t lineCount(const std::string &s)
{
    return static_cast<size_t>(std::count(s.begin(), s.end(), '\n'));
}
} // namespace

TEST(save_xyz, writes_full_double_precision)
{
    // GIVEN: a point with more than 6 significant digits
    const Eigen::Vector3d p(1234.567891234, -2345.678912345, 98.7654321);

    // WHEN: we write it as XYZ
    std::ostringstream out;
    ASSERT_TRUE(toXYZ(surfacesWith({p}), out));

    // THEN: it reads back exactly
    std::istringstream in(out.str());
    Eigen::Vector3d read;
    char comma;
    in >> read.x() >> comma >> read.y() >> comma >> read.z();
    EXPECT_EQ(read, p);
}

TEST(save_xyz, outlier_bounds_keep_points_within_one_metre)
{
    // GIVEN: points spread along x but within the same metre on y and z
    point_cloud cloud;
    for (int i = 0; i < 100; i++)
        cloud.emplace_back(i * 0.1, 20.2 + i * 0.003, 5.5 + i * 0.001);
    const auto surfaces = surfacesWith(cloud);

    // WHEN: we filter outliers and write the remaining points
    std::ostringstream out;
    ASSERT_TRUE(toXYZ(surfaces, out, filterOutliers(surfaces)));

    // THEN: every point is kept
    EXPECT_EQ(lineCount(out.str()), cloud.size());
}

TEST(save_xyz, outlier_bounds_drop_far_points)
{
    // GIVEN: a tight cluster with one distant outlier on each side
    point_cloud cloud;
    for (int i = 0; i < 100; i++)
        cloud.emplace_back(i % 10, i / 10, 0.5);
    cloud.emplace_back(-1000, 5, 0.5);
    cloud.emplace_back(1000, 5, 0.5);
    const auto surfaces = surfacesWith(cloud);

    // WHEN: we filter outliers and write the remaining points
    std::ostringstream out;
    ASSERT_TRUE(toXYZ(surfaces, out, filterOutliers(surfaces)));

    // THEN: only the cluster remains
    EXPECT_EQ(lineCount(out.str()), 100u);
}
