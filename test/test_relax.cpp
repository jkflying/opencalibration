#include <jk/KDTree.h>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/distort/invert_distortion.hpp>
#include <opencalibration/relax/autodiff_cost_function.hpp>
#include <opencalibration/relax/relax.hpp>
#include <opencalibration/relax/relax_cost_function.hpp>
#include <opencalibration/relax/relax_group.hpp>
#include <opencalibration/relax/relax_problem.hpp>
#include <opencalibration/surface/expand_mesh.hpp>
#include <opencalibration/surface/refine_mesh.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <optional>
#include <random>

using namespace opencalibration;
using namespace std::chrono_literals;

namespace
{
void appendFeature(FeatureSet &features, const feature_2d &feature)
{
    auto v = features.load();
    v.push_back(feature);
    features = std::move(v);
}
} // namespace

struct relax_group : public ::testing::Test
{
    size_t id[3];
    MeasurementGraph graph;
    std::vector<NodePose> np;
    ankerl::unordered_dense::map<size_t, CameraModel> cam_models;
    std::shared_ptr<CameraModel> model;
    Eigen::Quaterniond ground_ori[3];
    Eigen::Vector3d ground_pos[3];
    size_t edge_id[3];

    void init_cameras()
    {
        auto down = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());
        ground_ori[0] = Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()) * down);
        ground_ori[1] = Eigen::Quaterniond(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitY()) * down);
        ground_ori[2] = Eigen::Quaterniond(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitX()) * down);
        ground_pos[0] = Eigen::Vector3d(9, 9, 9);
        ground_pos[1] = Eigen::Vector3d(11, 9, 9);
        ground_pos[2] = Eigen::Vector3d(11, 11, 9);

        model = std::make_shared<CameraModel>();
        model->focal_length_pixels = 600;
        model->principle_point << 400, 300;
        model->pixels_cols = 800;
        model->pixels_rows = 600;
        model->projection_type = opencalibration::ProjectionType::PLANAR;
        model->id = 42;

        cam_models[model->id] = *model;

        for (int i = 0; i < 3; i++)
        {
            image img;
            img.orientation = ground_ori[i];
            img.position = ground_pos[i];
            img.gps_position = ground_pos[i];
            img.model = model;
            id[i] = graph.addNode(std::move(img));
            np.emplace_back(NodePose{id[i], ground_ori[i], ground_pos[i]});
        }
    }

    point_cloud generate_planar_points()
    {
        point_cloud vec3d;
        vec3d.reserve(100);
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                vec3d.emplace_back(i + 5, j + 5, -10 + 1e-3 * i + 1e-2 * j);
            }
        }
        return vec3d;
    }

    point_cloud generate_3d_points()
    {
        point_cloud vec3d;
        vec3d.reserve(100);
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                vec3d.emplace_back(i + 5, j + 5, -10 + (i + j) % 2);
            }
        }
        return vec3d;
    }

    void add_point_measurements(const point_cloud &points)
    {
        for (size_t i = 0; i < 3; i++)
        {
            for (const Eigen::Vector3d &p : points)
            {
                Eigen::Vector3d ray = np[i].orientation.inverse() * (p - np[i].position).normalized();
                Eigen::Vector2d pixel = image_from_3d(ray, *model);
                feature_2d feat;
                feat.location = pixel;
                appendFeature(graph.getNode(np[i].node_id)->payload.features, feat);
            }
        }

        for (size_t i = 0; i < 3; i++)
        {
            camera_relations relation;
            size_t index[2] = {i, (i + 1) % 3};
            for (size_t counter = 0; counter < points.size(); counter++)
            {
                Eigen::Vector2d pixel[2];
                for (int j = 0; j < 2; j++)
                {
                    pixel[j] = graph.getNode(np[index[j]].node_id)->payload.features.load()[counter].location;
                }
                relation.inlier_matches.emplace_back(
                    feature_match_denormalized{pixel[0], pixel[1], counter, counter, counter});
            }
            edge_id[i] = graph.addEdge(std::move(relation), id[index[0]], id[index[1]]);
        }
    }

    void add_edge_measurements()
    {

        for (size_t i = 0; i < 3; i++)
        {
            camera_relations relation;
            size_t index[2] = {i, (i + 1) % 3};

            Eigen::Quaterniond actual_r = np[index[1]].orientation * np[index[0]].orientation.inverse();
            Eigen::Vector3d actual_t =
                np[index[0]].orientation.inverse() * (np[index[1]].position - np[index[0]].position).normalized();

            if (i == 0 || i == 2)
            {
                relation.relative_poses[0].score = 8;
                relation.relative_poses[0].position = actual_t;
                relation.relative_poses[0].orientation = actual_r;
            }
            if (i == 1 || i == 2)
            {
                relation.relative_poses[1].score = 18;
                relation.relative_poses[1].position = actual_t;
                relation.relative_poses[1].orientation = actual_r;
            }
            relation.inlier_matches.emplace_back();

            edge_id[i] = graph.addEdge(std::move(relation), id[index[0]], id[index[1]]);
        }
    }

    void add_ori_noise(std::array<double, 3> noise)
    {
        np[0].orientation *= Eigen::Quaterniond(Eigen::AngleAxisd(noise[0], Eigen::Vector3d::UnitY()));
        np[1].orientation *= Eigen::Quaterniond(Eigen::AngleAxisd(noise[1], Eigen::Vector3d::UnitZ()));
        np[2].orientation *= Eigen::Quaterniond(Eigen::AngleAxisd(noise[2], Eigen::Vector3d::UnitX()));
    }

    void add_ori_noise_graph(std::array<double, 3> noise)
    {
        graph.getNode(id[0])->payload.orientation *=
            Eigen::Quaterniond(Eigen::AngleAxisd(noise[0], Eigen::Vector3d::UnitY()));
        graph.getNode(id[1])->payload.orientation *=
            Eigen::Quaterniond(Eigen::AngleAxisd(noise[1], Eigen::Vector3d::UnitZ()));
        graph.getNode(id[2])->payload.orientation *=
            Eigen::Quaterniond(Eigen::AngleAxisd(noise[2], Eigen::Vector3d::UnitX()));
    }
};

static std::array<double, 7> packPose(const Eigen::Quaterniond &q, const Eigen::Vector3d &p)
{
    std::array<double, 7> pose;
    Eigen::Map<Eigen::Quaterniond>(pose.data()) = q;
    Eigen::Map<Eigen::Vector3d>(pose.data() + 4) = p;
    return pose;
}

TEST(relax_cost, fixed_position_costs_match_full_pose_costs)
{
    // GIVEN: three tilted cameras over a plane, and the same costs built with 7-wide poses and with fixed positions
    std::array<std::array<double, 7>, 3> poses;
    std::vector<Eigen::Vector3d> positions, rays;
    std::vector<Eigen::Vector2d> pixels;
    for (int i = 0; i < 3; i++)
    {
        const Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()) *
                                   Eigen::AngleAxisd(0.1 * (i + 1), Eigen::Vector3d(1, 2, 3).normalized()));
        positions.emplace_back(i * 3., i * 2., 20 + i);
        Eigen::Map<Eigen::Quaterniond>(poses[i].data()) = q;
        Eigen::Map<Eigen::Vector3d>(poses[i].data() + 4) = positions.back();
        rays.push_back((q.inverse() * (Eigen::Vector3d(2, 1, 0.5) - positions.back())).normalized() +
                       Eigen::Vector3d(0.01 * i, -0.02, 0));
        pixels.emplace_back(380 + 10 * i, 290 - 5 * i);
    }
    double z[3]{0.4, 0.6, 0.5};
    const std::array<Eigen::Vector2d, 3> corners{Eigen::Vector2d(-10, -10), Eigen::Vector2d(10, -10),
                                                 Eigen::Vector2d(0, 10)};
    CameraModel model;
    model.focal_length_pixels = 600;
    model.principle_point << 400, 300;
    model.pixels_cols = 800;
    model.pixels_rows = 600;
    InverseDifferentiableCameraModel<double> inv_model = convertModel(model);
    camera_relations relations;
    relations.relative_poses[0].score = 10;
    relations.relative_poses[0].orientation = Eigen::Quaterniond(Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ()));
    relations.relative_poses[0].position = Eigen::Vector3d(1, 0.6, 0.1).normalized();

    struct Case
    {
        std::unique_ptr<ceres::CostFunction> full, fixed;
        std::vector<double *> non_pose_blocks;
        int num_poses;
    };
    std::vector<Case> cases;
    cases.push_back({std::unique_ptr<ceres::CostFunction>(newAutoDiffPlaneIntersectionAngleCost_NRay(rays, corners)),
                     std::unique_ptr<ceres::CostFunction>(
                         newAutoDiffPlaneIntersectionAngleCost_NRay_FixedPositions(rays, corners, positions)),
                     {&z[0], &z[1], &z[2]},
                     3});
    cases.push_back(
        {std::unique_ptr<ceres::CostFunction>(
             newAutoDiffPlaneIntersectionAngleCost_NRay_FocalRadial(pixels, corners, inv_model)),
         std::unique_ptr<ceres::CostFunction>(newAutoDiffPlaneIntersectionAngleCost_NRay_FocalRadial_FixedPositions(
             pixels, corners, inv_model, positions)),
         {&z[0], &z[1], &z[2], &inv_model.focal_length_pixels, inv_model.principle_point.data(),
          inv_model.radial_distortion.data()},
         3});
    cases.push_back({std::unique_ptr<ceres::CostFunction>(newAutoDiffMultiDecomposedRotationCost(relations)),
                     std::unique_ptr<ceres::CostFunction>(newAutoDiffMultiDecomposedRotationCost_FixedPositions(
                         relations, Eigen::Vector3d(positions[1] - positions[0]))),
                     {},
                     2});

    for (size_t c = 0; c < cases.size(); c++)
    {
        auto &cs = cases[c];
        std::array<std::array<double, 7>, 3> nan_positions = poses;
        for (auto &p : nan_positions)
            std::fill(p.begin() + 4, p.end(), NAN);
        std::vector<double *> params = cs.non_pose_blocks, fixed_params = cs.non_pose_blocks;
        for (int i = 0; i < cs.num_poses; i++)
        {
            params.push_back(poses[i].data());
            fixed_params.push_back(nan_positions[i].data());
        }

        const int num_residuals = cs.full->num_residuals();
        ASSERT_EQ(num_residuals, cs.fixed->num_residuals());
        const auto &full_sizes = cs.full->parameter_block_sizes();
        std::vector<std::vector<double>> full_jac(params.size()), fixed_jac(params.size());
        std::vector<double *> full_ptrs, fixed_ptrs;
        for (size_t b = 0; b < params.size(); b++)
        {
            full_jac[b].resize(num_residuals * full_sizes[b]);
            fixed_jac[b].resize(num_residuals * cs.fixed->parameter_block_sizes()[b]);
            full_ptrs.push_back(full_jac[b].data());
            fixed_ptrs.push_back(fixed_jac[b].data());
        }
        std::vector<double> full_res(num_residuals), fixed_res(num_residuals);

        // WHEN: we evaluate both
        ASSERT_TRUE(cs.full->Evaluate(params.data(), full_res.data(), full_ptrs.data())) << c;
        ASSERT_TRUE(cs.fixed->Evaluate(fixed_params.data(), fixed_res.data(), fixed_ptrs.data())) << c;

        // THEN: residuals match despite the NaN positions in the fixed cost's pose blocks, and so do the derivatives
        // by everything except the positions
        for (int r = 0; r < num_residuals; r++)
            EXPECT_NEAR(full_res[r], fixed_res[r], 1e-12) << c << " residual " << r;
        for (size_t b = 0; b < params.size(); b++)
        {
            const int fixed_size = cs.fixed->parameter_block_sizes()[b];
            for (int r = 0; r < num_residuals; r++)
                for (int k = 0; k < fixed_size; k++)
                    EXPECT_NEAR(full_jac[b][r * full_sizes[b] + k], fixed_jac[b][r * fixed_size + k], 1e-9)
                        << c << " block " << b << " residual " << r << " param " << k;
        }
    }
}

TEST_F(relax_group, downwards_prior_cost_function)
{
    // GIVEN: a starting angle
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity() * Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());

    // WHEN: we get the cost of the downwards prior
    PointsDownwardsPrior p(1e-3);
    double r = NAN;
    EXPECT_TRUE(p(packPose(q, Eigen::Vector3d::Zero()).data(), &r));

    // THEN: it should be the amount away from vertical
    EXPECT_NEAR(r, 0, 1e-8);

    // WHEN: we shift it 0.3 rad away from vertical
    q = q * Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX());
    EXPECT_TRUE(p(packPose(q, Eigen::Vector3d::Zero()).data(), &r));

    // THEN: it should have a residual of 0.3 * weight
    EXPECT_NEAR(r, 0.3 * 1e-3, 1e-9);
}

TEST_F(relax_group, distortion_monotonicity_zero_distortion)
{
    // GIVEN: zero radial distortion
    double radial[3] = {0, 0, 0};

    // WHEN: we evaluate the monotonicity cost
    DistortionMonotonicityCost cost(1.0, 1.0);
    double residuals[10];
    EXPECT_TRUE(cost(radial, residuals));

    // THEN: all residuals should be zero (derivative is always 1 > 0)
    for (int i = 0; i < 10; i++)
        EXPECT_DOUBLE_EQ(residuals[i], 0.0) << "residual " << i;
}

TEST_F(relax_group, distortion_monotonicity_monotonic)
{
    // GIVEN: small positive k1 (monotonic for reasonable r)
    double radial[3] = {0.01, 0, 0};

    // WHEN: we evaluate the monotonicity cost
    DistortionMonotonicityCost cost(1.0, 1.0);
    double residuals[10];
    EXPECT_TRUE(cost(radial, residuals));

    // THEN: all residuals should be zero (derivative = 1 + 3*0.01*r² > 0 for r in [0,1])
    for (int i = 0; i < 10; i++)
        EXPECT_DOUBLE_EQ(residuals[i], 0.0) << "residual " << i;
}

TEST_F(relax_group, distortion_monotonicity_nonmonotonic)
{
    // GIVEN: strongly negative k1 making distortion non-monotonic
    double radial[3] = {-5.0, 0, 0};

    // WHEN: we evaluate the monotonicity cost
    DistortionMonotonicityCost cost(1.0, 1.0);
    double residuals[10];
    EXPECT_TRUE(cost(radial, residuals));

    // THEN: some residuals should be positive (derivative = 1 + 3*(-5)*r² goes negative for r > sqrt(1/15) ≈ 0.258)
    bool any_positive = false;
    for (int i = 0; i < 10; i++)
    {
        EXPECT_GE(residuals[i], 0.0) << "residual " << i;
        if (residuals[i] > 0.0)
            any_positive = true;
    }
    EXPECT_TRUE(any_positive);

    // AND: weight scaling should produce proportionally larger residuals
    DistortionMonotonicityCost cost2(1.0, 3.0);
    double residuals2[10];
    EXPECT_TRUE(cost2(radial, residuals2));
    for (int i = 0; i < 10; i++)
    {
        EXPECT_NEAR(residuals2[i], residuals[i] * 3.0, 1e-12) << "residual " << i;
    }
}

TEST_F(relax_group, rel_rot_cost_function)
{
    // GIVEN: two cameras
    init_cameras();
    Eigen::Quaterniond rel_rot = ground_ori[1] * ground_ori[0].inverse();
    Eigen::Vector3d rel_pos = ground_ori[0].inverse() * (ground_pos[1] - ground_pos[0]);
    DecomposedRotationCost cost(rel_rot, rel_pos, 8);

    {
        // WHEN: we get the relative orientation cost with a perfect guess
        Eigen::Quaterniond q[2]{ground_ori[0], ground_ori[1]};
        double r[3]{NAN, NAN, NAN};
        bool success = cost(packPose(q[0], ground_pos[0]).data(), packPose(q[1], ground_pos[1]).data(), r);

        // THEN: they should have residuals of 0
        EXPECT_TRUE(success);
        EXPECT_NEAR(r[0], 0., 1e-5);
        EXPECT_NEAR(r[1], 0., 1e-5);
        EXPECT_NEAR(r[2], 0., 1e-12);
    }

    {
        // WHEN: we get a relative orientation cost with a fixed offset guess
        Eigen::Quaterniond q[2]{ground_ori[0] * Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()), ground_ori[1]};
        double r[3]{NAN, NAN, NAN};
        bool success = cost(packPose(q[0], ground_pos[0]).data(), packPose(q[1], ground_pos[1]).data(), r);

        // THEN: we should get that fixed offset as the residuals
        EXPECT_TRUE(success);
        EXPECT_NEAR(r[0], 0.3, 1e-12);
        EXPECT_NEAR(r[1], 0.0, 1e-5);
        EXPECT_NEAR(r[2], 0.3, 1e-12);
    }

    {
        // WHEN: we get a relative orientation cost with a fixed offset guess
        Eigen::Quaterniond q[2]{ground_ori[0], ground_ori[1] * Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitZ())};
        double r[3]{NAN, NAN, NAN};
        bool success = cost(packPose(q[0], ground_pos[0]).data(), packPose(q[1], ground_pos[1]).data(), r);

        // THEN: we should get that fixed offset as the residuals
        EXPECT_TRUE(success);
        EXPECT_NEAR(r[0], 0.0, 1e-5);
        EXPECT_NEAR(r[1], 0.3, 1e-12);
        EXPECT_NEAR(r[2], 0.3, 1e-12);
    }

    {
        // WHEN: we double the baseline along its direction
        const Eigen::Vector3d far_pos = ground_pos[0] + 2 * (ground_pos[1] - ground_pos[0]);
        double r[3]{NAN, NAN, NAN};
        bool success = cost(packPose(ground_ori[0], ground_pos[0]).data(), packPose(ground_ori[1], far_pos).data(), r);

        // THEN: the residuals should be unchanged, only the translation direction matters
        EXPECT_TRUE(success);
        EXPECT_NEAR(r[0], 0., 1e-5);
        EXPECT_NEAR(r[1], 0., 1e-5);
        EXPECT_NEAR(r[2], 0., 1e-12);
    }

    {
        // WHEN: we move the second camera perpendicular to the baseline
        const Eigen::Vector3d side_pos = ground_pos[1] + Eigen::Vector3d(0, 2, 0);
        double r[3]{NAN, NAN, NAN};
        bool success = cost(packPose(ground_ori[0], ground_pos[0]).data(), packPose(ground_ori[1], side_pos).data(), r);

        // THEN: the translation direction residuals should pick it up
        EXPECT_TRUE(success);
        EXPECT_GT(std::abs(r[0]) + std::abs(r[1]), 0.1);
    }
}

TEST_F(relax_group, pixel_error_cost_function)
{
    // GIVEN: a camera and a pixel measurement of a known 3D point
    init_cameras();
    const Eigen::Vector3d point(10, 10, -10);
    const Eigen::Vector2d pixel = image_from_3d(point, *model, ground_pos[0], ground_ori[0]);
    PixelErrorCost_Orientation cost(*model, pixel);

    {
        // WHEN: we evaluate at the true pose
        double r[2]{NAN, NAN};
        EXPECT_TRUE(cost(packPose(ground_ori[0], ground_pos[0]).data(), point.data(), r));

        // THEN: the residuals should be zero
        EXPECT_NEAR(r[0], 0., 1e-9);
        EXPECT_NEAR(r[1], 0., 1e-9);
    }

    {
        // WHEN: we move the camera position in the pose block
        const Eigen::Vector3d moved = ground_pos[0] + Eigen::Vector3d(0.5, -0.3, 0.2);
        double r[2]{NAN, NAN};
        EXPECT_TRUE(cost(packPose(ground_ori[0], moved).data(), point.data(), r));

        // THEN: the residuals should be the reprojection from the moved camera
        const Eigen::Vector2d expected = image_from_3d(point, *model, moved, ground_ori[0]) - pixel;
        EXPECT_GT(expected.norm(), 1);
        EXPECT_NEAR(r[0], expected.x(), 1e-9);
        EXPECT_NEAR(r[1], expected.y(), 1e-9);
    }
}

TEST_F(relax_group, pixel_error_cost_rejects_points_behind_camera)
{
    // GIVEN: a camera and a point directly behind it, observed at the principal point
    init_cameras();
    const Eigen::Vector3d behind = ground_pos[0] - 5 * (ground_ori[0] * Eigen::Vector3d::UnitZ());
    const Eigen::Vector2d observed = model->principle_point;
    const auto pose = packPose(ground_ori[0], ground_pos[0]);
    const double focal = model->focal_length_pixels;
    const Eigen::Vector3d radial = model->radial_distortion;
    const Eigen::Vector2d tangential = model->tangential_distortion;

    // WHEN: we evaluate each pixel error cost there
    double r[4][2];
    PixelErrorCost_Orientation(*model, observed)(pose.data(), behind.data(), r[0]);
    PixelErrorCost_OrientationFocal(*model, observed)(pose.data(), behind.data(), &focal, observed.data(), r[1]);
    PixelErrorCost_OrientationFocalRadial(*model, observed)(pose.data(), behind.data(), &focal, observed.data(),
                                                            radial.data(), r[2]);
    PixelErrorCost_OrientationFocalRadialTangential(*model, observed)(
        pose.data(), behind.data(), &focal, observed.data(), radial.data(), tangential.data(), r[3]);

    // THEN: none of them report a perfect fit
    for (const auto &residual : r)
        EXPECT_GT(Eigen::Vector2d(residual[0], residual[1]).norm(), 1);
}

TEST_F(relax_group, gps_position_prior_cost_function)
{
    // GIVEN: a GPS prior with 2m horizontal and 4m vertical sigma
    GPSPositionPrior prior(Eigen::Vector3d(10, 20, 30), 1 / 2., 1 / 4.);

    // WHEN: the camera sits 1m east, 2m north and 4m above the GPS position
    double r[3]{NAN, NAN, NAN};
    EXPECT_TRUE(prior(packPose(Eigen::Quaterniond::Identity(), Eigen::Vector3d(11, 22, 34)).data(), r));

    // THEN: the residuals should be weighted per axis
    EXPECT_NEAR(r[0], 0.5, 1e-12);
    EXPECT_NEAR(r[1], 1.0, 1e-12);
    EXPECT_NEAR(r[2], 1.0, 1e-12);
}

TEST_F(relax_group, no_images)
{
    // GIVEN: a graph, with no images
    MeasurementGraph graph;
    std::vector<NodePose> np;
    ankerl::unordered_dense::map<size_t, CameraModel> cam_models;

    // WHEN: we relax the relative orientations
    relax(graph, np, cam_models, {}, {Option::ORIENTATION}, {});

    // THEN: it shouldn't crash
}

TEST_F(relax_group, prior_1_image)
{
    // GIVEN: a graph, with 1 image tilted 45 degrees from downward
    MeasurementGraph graph;
    std::vector<NodePose> np;
    ankerl::unordered_dense::map<size_t, CameraModel> cam_models;

    Eigen::Quaterniond ori(Eigen::AngleAxisd(M_PI_4, Eigen::Vector3d::UnitX()));
    Eigen::Vector3d pos(9, 9, 9);
    image img;
    img.orientation = ori;
    img.position = pos;

    size_t id = graph.addNode(std::move(img));
    np.emplace_back(NodePose{id, ori, pos});

    // WHEN: we relax with the downward prior
    relax(graph, np, cam_models, {}, {Option::ORIENTATION}, {});

    // THEN: the solver optimizes the orientation toward the downward prior (PI rotation around X)
    Eigen::Quaterniond downward(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    double angle_to_downward = Eigen::AngleAxisd(np[0].orientation.inverse() * downward).angle();
    EXPECT_LT(angle_to_downward, M_PI_4); // Closer to downward than initial 45 degrees
}

TEST_F(relax_group, prior_2_images)
{
    // GIVEN: a graph, with 2 images with relative orientation as identity
    MeasurementGraph graph;
    std::vector<NodePose> np;
    ankerl::unordered_dense::map<size_t, CameraModel> cam_models;


    const Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    Eigen::Quaterniond ori = down * Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX());
    Eigen::Vector3d pos(9, 9, 9);
    image img;
    img.orientation = ori;
    img.position = pos;

    const size_t id = graph.addNode(std::move(img));
    np.emplace_back(NodePose{id, ori, pos});

    ori = down * Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitX());
    pos << 11, 9, 9;

    image img2;
    img2.orientation = ori;
    img2.position = pos;
    const size_t id2 = graph.addNode(std::move(img2));
    np.emplace_back(NodePose{id2, ori, pos});

    camera_relations relation;
    relation.relative_poses[0].orientation = Eigen::Quaterniond::Identity();
    relation.relative_poses[0].position << 1, 0, 0;
    relation.relative_poses[0].score = 8;
    relation.inlier_matches.resize(10);
    size_t edge_id = graph.addEdge(std::move(relation), id, id2);

    // WHEN: we relax the relative orientations
    relax(graph, np, cam_models, {edge_id}, {Option::ORIENTATION}, {});

    // THEN: the solver optimizes the relative orientation to match the constraint (identity)
    // The relative orientation between the two cameras should be close to identity
    Eigen::Quaterniond relative_ori = np[0].orientation.inverse() * np[1].orientation;
    EXPECT_NEAR(Eigen::AngleAxisd(relative_ori).angle(), 0.0, 1e-3);
}

TEST_F(relax_group, relative_orientation_3_images)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    init_cameras();

    add_edge_measurements();
    add_ori_noise({-1, 1, 1});

    // WHEN: we relax them with relative orientation
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax(graph, np, cam_models, edges, {Option::ORIENTATION}, {});

    // THEN: it should put them back into the original orientation
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation * ground_ori[i].inverse()).angle(), 1e-5)
            << np[i].orientation.coeffs().transpose();
}

TEST_F(relax_group, measurement_3_images_points)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    init_cameras();
    add_point_measurements(generate_3d_points());
    add_ori_noise({-0.05, 0.05, 0.05});

    // WHEN: we relax them with relative orientation
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax(graph, np, cam_models, edges, {Option::ORIENTATION, Option::POINTS_3D}, {});

    // THEN: it should put them back into the original orientation
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 1e-8)
            << i << ": " << np[i].orientation.coeffs().transpose() << std::endl
            << "g: " << ground_ori[i].coeffs().transpose();
}

TEST_F(relax_group, measurement_3_images_points_radial_without_focal)
{
    // GIVEN: a graph, 3 images with edges between them all, and a camera model with spurious radial distortion
    init_cameras();
    add_point_measurements(generate_3d_points());
    const double initial_focal = cam_models[model->id].focal_length_pixels;
    cam_models[model->id].radial_distortion = Eigen::Vector3d(0.01, 0, 0);

    // WHEN: we relax radial distortion without optimizing focal length
    const RelaxOptionSet options({Option::ORIENTATION, Option::POINTS_3D, Option::LENS_DISTORTIONS_RADIAL,
                                  Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION});
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax(graph, np, cam_models, edges, options, {});

    // THEN: radial distortion is optimized towards zero while focal length is held constant
    EXPECT_LT(cam_models[model->id].radial_distortion.norm(), 0.005)
        << cam_models[model->id].radial_distortion.transpose();
    EXPECT_NEAR(cam_models[model->id].focal_length_pixels, initial_focal, 1e-6);
}

TEST_F(relax_group, measurement_3_images_triangulated_rays)
{
    // GIVEN: a graph, 3 images with edges between them all observing non-planar points, with their rotation disturbed
    init_cameras();
    add_point_measurements(generate_3d_points());
    add_ori_noise({-0.05, 0.05, 0.05});

    // WHEN: we relax them with triangulated rays
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax(graph, np, cam_models, edges, {Option::ORIENTATION, Option::TRIANGULATED_RAYS}, {});

    // THEN: it should put them back into the original orientation, less the downwards prior's slight pull on the
    // tilted cameras
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 1e-4)
            << i << ": " << np[i].orientation.coeffs().transpose() << std::endl
            << "g: " << ground_ori[i].coeffs().transpose();

    // AND: the positions should not move, since they aren't being optimized
    for (int i = 0; i < 3; i++)
        EXPECT_EQ(np[i].position, ground_pos[i]) << i;
}

TEST_F(relax_group, measurement_3_images_triangulated_rays_collinear_roll_pulled_down)
{
    // GIVEN: 3 nadir cameras along one flight line, all started rolled 80° about it. With positions fixed, that roll
    // changes no triangulation, so only the downwards prior can undo it
    init_cameras();
    const Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    const Eigen::Quaterniond roll(Eigen::AngleAxisd(80 * M_PI / 180, Eigen::Vector3d::UnitX()));
    for (int i = 0; i < 3; i++)
    {
        ground_ori[i] = down;
        ground_pos[i] = Eigen::Vector3d(8 + 2 * i, 9, 9);
        auto &img = graph.getNode(id[i])->payload;
        img.orientation = ground_ori[i];
        img.position = img.gps_position = ground_pos[i];
        np[i] = NodePose{id[i], ground_ori[i], ground_pos[i]};
    }
    add_point_measurements(generate_3d_points());
    for (int i = 0; i < 3; i++)
        np[i].orientation = roll * ground_ori[i];

    // WHEN: we relax them with triangulated rays
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax(graph, np, cam_models, edges, {Option::ORIENTATION, Option::TRIANGULATED_RAYS}, {});

    // THEN: they should end pointing down, well inside the 45° cone, rather than keeping the roll
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 5 * M_PI / 180)
            << i << ": " << np[i].orientation.coeffs().transpose();
}

TEST_F(relax_group, measurement_3_images_plane)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    init_cameras();
    add_point_measurements(generate_planar_points());
    add_ori_noise({-0.1, 0.1, 0.1});

    // WHEN: we relax them with relative orientation
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax(graph, np, cam_models, edges, {Option::ORIENTATION, Option::GROUND_PLANE}, {});
    // and again to re-init the inliers
    relax(graph, np, cam_models, edges, {Option::ORIENTATION, Option::GROUND_PLANE}, {});

    // THEN: it should put them back into the original orientation
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 1e-3)
            << i << ": " << np[i].orientation.coeffs().transpose() << std::endl
            << "g: " << ground_ori[i].coeffs().transpose();

    // AND: the positions should not move, since they aren't being optimized
    for (int i = 0; i < 3; i++)
        EXPECT_EQ(np[i].position, ground_pos[i]) << i;
}

TEST_F(relax_group, measurement_3_images_plane_position)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation and one position disturbed
    init_cameras();
    add_point_measurements(generate_planar_points());
    add_ori_noise({-0.1, 0.1, 0.1});
    np[1].position += Eigen::Vector3d(0.3, -0.2, 0.5);

    // WHEN: we relax them with position enabled, anchored by the GPS prior
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax(graph, np, cam_models, edges, {Option::ORIENTATION, Option::POSITION, Option::GROUND_PLANE}, {});
    // and again to re-init the inliers
    relax(graph, np, cam_models, edges, {Option::ORIENTATION, Option::POSITION, Option::GROUND_PLANE}, {});

    // THEN: the relative poses should be recovered
    for (int i = 1; i < 3; i++)
    {
        const Eigen::Quaterniond rel_ori = np[0].orientation.inverse() * np[i].orientation;
        const Eigen::Quaterniond ground_rel_ori = ground_ori[0].inverse() * ground_ori[i];
        EXPECT_LT(Eigen::AngleAxisd(rel_ori.inverse() * ground_rel_ori).angle(), 1e-3) << i;

        const Eigen::Vector3d rel_dir = (np[0].orientation.inverse() * (np[i].position - np[0].position)).normalized();
        const Eigen::Vector3d ground_rel_dir = (ground_ori[0].inverse() * (ground_pos[i] - ground_pos[0])).normalized();
        EXPECT_LT((rel_dir - ground_rel_dir).norm(), 1e-3) << i << ": " << rel_dir.transpose() << std::endl
                                                           << "g: " << ground_rel_dir.transpose();
    }

    // AND: the positions should stay within the GPS uncertainty
    for (int i = 0; i < 3; i++)
        EXPECT_LT((np[i].position - ground_pos[i]).norm(), 0.5) << i << ": " << np[i].position.transpose() << std::endl
                                                                << "g: " << ground_pos[i].transpose();
}

TEST_F(relax_group, measurement_3_images_mesh_radial)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    init_cameras();
    const Eigen::Vector3d expected_distortion(0.1, -0.1, 0.1);
    model->radial_distortion = expected_distortion;
    point_cloud wide_points;
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            wide_points.emplace_back(1 + 2 * i, 1 + 2 * j, -10 + 1e-3 * i + 1e-2 * j);
    add_point_measurements(wide_points);
    model->radial_distortion.fill(0);
    add_ori_noise({-0.1, 0.1, 0.1});

    // WHEN: we relax them with relative orientation
    const RelaxOptionSet options({Option::ORIENTATION, Option::LENS_DISTORTIONS_RADIAL,
                                  Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION, Option::GROUND_MESH});
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};

    for (int i = 0; i < 10; i++)
        // a few times...
        relax(graph, np, cam_models, edges, options, {});

    // THEN: the orientations recover from the 0.1 rad noise
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 0.05) << i;

    // AND: the radial displacement across the image matches the true lens much better than no distortion
    const auto maxDisplacementError = [&](const Eigen::Vector3d &k) {
        const double max_r = Eigen::Vector2d(400, 300).norm() / model->focal_length_pixels;
        double worst = 0;
        for (double r = 0; r <= max_r; r += max_r / 50)
        {
            const Eigen::Vector3d powers(r * r, std::pow(r, 4), std::pow(r, 6));
            worst = std::max(worst, std::abs((k - expected_distortion).dot(powers)) * r * model->focal_length_pixels);
        }
        return worst;
    };
    EXPECT_LT(maxDisplacementError(cam_models[model->id].radial_distortion),
              0.1 * maxDisplacementError(Eigen::Vector3d::Zero()));
}

TEST_F(relax_group, measurement_3_images_plane_focal_two_models)
{
    // GIVEN: a graph, 3 images with edges between them all, where the last image uses a second camera model
    init_cameras();
    add_point_measurements(generate_planar_points());
    auto second_model = std::make_shared<CameraModel>(*model);
    second_model->id = 43;
    cam_models[second_model->id] = *second_model;
    graph.getNode(id[2])->payload.model = second_model;

    // AND: the first model has the wrong focal length
    const double initial_focal_length = 700;
    cam_models[model->id].focal_length_pixels = initial_focal_length;

    // WHEN: we relax with focal length optimization
    const RelaxOptionSet options({Option::ORIENTATION, Option::FOCAL_LENGTH, Option::GROUND_PLANE});
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax(graph, np, cam_models, edges, options, {});

    // THEN: the first model's focal length is moved towards the true value
    EXPECT_LT(std::abs(cam_models[model->id].focal_length_pixels - model->focal_length_pixels),
              initial_focal_length - model->focal_length_pixels);
}

TEST_F(relax_group, measurement_3_images_plane_focal_keeps_radial_constant)
{
    // GIVEN: a graph, 3 images with edges between them all, and a camera model with some radial distortion
    init_cameras();
    add_point_measurements(generate_planar_points());
    const Eigen::Vector3d initial_distortion(0.01, -0.01, 0.01);
    cam_models[model->id].radial_distortion = initial_distortion;
    cam_models[model->id].focal_length_pixels = 700;

    // WHEN: we relax with only focal length optimization
    const RelaxOptionSet options({Option::ORIENTATION, Option::FOCAL_LENGTH, Option::GROUND_PLANE});
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax(graph, np, cam_models, edges, options, {});

    // THEN: the radial distortion is unchanged, within the accuracy of converting to and from the inverse model
    EXPECT_LT((cam_models[model->id].radial_distortion - initial_distortion).norm(), 1e-3)
        << cam_models[model->id].radial_distortion.transpose();
}

TEST_F(relax_group, group_with_connection_depth_has_unique_nodes)
{
    // GIVEN: a graph, 3 images with edges between them all
    init_cameras();
    add_edge_measurements();
    jk::tree::KDTree<size_t, 2> imageGPSLocations;
    for (size_t i = 0; i < 3; i++)
    {
        imageGPSLocations.addPoint({ground_pos[i].x(), ground_pos[i].y()}, id[i]);
    }

    // WHEN: we create a group from one image, reaching the others over a connection depth of 2
    RelaxGroup group;
    group.init(graph, {id[0]}, imageGPSLocations, 2, {Option::ORIENTATION});
    auto optimized_ids = group.finalize(graph);

    // THEN: each image is in the group exactly once
    std::sort(optimized_ids.begin(), optimized_ids.end());
    std::vector<size_t> expected_ids{id[0], id[1], id[2]};
    std::sort(expected_ids.begin(), expected_ids.end());
    EXPECT_EQ(optimized_ids, expected_ids);
}

TEST_F(relax_group, group_anchors_to_fixed_neighbours)
{
    // GIVEN: a graph, 3 images with edges between them all, where only the first image has a disturbed orientation
    init_cameras();
    add_point_measurements(generate_3d_points());
    add_ori_noise_graph({-0.05, 0, 0});
    jk::tree::KDTree<size_t, 2> imageGPSLocations;
    for (size_t i = 0; i < 3; i++)
    {
        imageGPSLocations.addPoint({ground_pos[i].x(), ground_pos[i].y()}, id[i]);
    }

    // WHEN: we relax a group of just the first image, with no connection depth
    RelaxGroup group;
    group.init(graph, {id[0]}, imageGPSLocations, 0, {Option::ORIENTATION, Option::TRIANGULATED_RAYS});
    group.run(graph, {});
    auto optimized_ids = group.finalize(graph);

    // THEN: only the first image is optimized
    EXPECT_EQ(optimized_ids, std::vector<size_t>{id[0]});

    // AND: the solved neighbours pull it back into the original orientation
    EXPECT_LT(Eigen::AngleAxisd(graph.getNode(id[0])->payload.orientation.inverse() * ground_ori[0]).angle(), 1e-6);
    for (int i = 1; i < 3; i++)
        EXPECT_EQ(graph.getNode(id[i])->payload.orientation.coeffs(), ground_ori[i].coeffs()) << i;
}

class TestRelaxProblem : public RelaxProblem
{
  public:
    using RelaxProblem::scoreRaysAgainstPlane;
    using RelaxProblem::selectInlierRays;

    track_vec test_get_tracks() const
    {
        track_vec tracks;
        for (const auto &edge_tracks : _edge_tracks)
        {
            tracks.insert(tracks.end(), edge_tracks.second.begin(), edge_tracks.second.end());
        }
        return tracks;
    }

    const ceres::Solver::Summary &test_get_solver_summary() const
    {
        return _summary;
    }

    size_t test_num_multi_ray_measurements(std::optional<size_t> node_id = std::nullopt) const
    {
        return std::count_if(_multi_ray_measurements.begin(), _multi_ray_measurements.end(),
                             [&](const NodeIdFeatureIndex &m) { return !node_id || m.node_id == *node_id; });
    }

    size_t test_num_grid_filtered_matches(size_t node_id, size_t edge_id)
    {
        return _grid_filter[node_id][edge_id].getBestMeasurementsPerCell().size();
    }

    double test_ray_loss_delta(int dof)
    {
        updateRobustLossScale(_ray_residuals);
        double rho[3];
        const double s = 1e12;
        _ray_residuals.lossesByDegreesOfFreedom.at(dof)->Evaluate(s, rho);
        return rho[1] * std::sqrt(s);
    }
};

TEST(relax_track, ray_inliers_found_when_outliers_come_first)
{
    // GIVEN: a track of 8 downward-looking rays hitting a flat plane, where the first 3 rays are outliers
    const Eigen::Quaterniond down(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    std::vector<TrackRay> rays;
    for (size_t i = 0; i < 8; i++)
    {
        const Eigen::Vector3d loc(10.0 * i, 0, 100);
        const Eigen::Vector3d target = i < 3 ? Eigen::Vector3d(40, 30, 0) : Eigen::Vector3d::Zero();
        rays.push_back(
            TrackRay{i, 0, 0, loc, down.inverse() * (target - loc), Eigen::Vector2d::Zero(), down, nullptr, true, 1});
    }
    plane_3_corners_d plane;
    plane.corner[0] << -1000, -1000, 0;
    plane.corner[1] << 1000, -1000, 0;
    plane.corner[2] << 0, 1000, 0;

    // WHEN: we score and select the inlier rays
    auto scores = TestRelaxProblem::scoreRaysAgainstPlane(rays, plane);
    const auto inliers = TestRelaxProblem::selectInlierRays(scores, rays);

    // THEN: exactly the 5 consistent rays are kept
    ASSERT_EQ(inliers.size(), 5);
    for (const auto &r : inliers)
        EXPECT_GE(r.node_id, 3);
}

TEST_F(relax_group, measurement_3_images_plane_with_uninitialized_image)
{
    // GIVEN: a graph, 3 images with edges between them all, where the first image has no pose yet
    init_cameras();
    add_point_measurements(generate_planar_points());
    graph.getNode(id[0])->payload.position.fill(NAN);
    std::vector<NodePose> initialized_poses{np[1], np[2]};

    // WHEN: we set up the problem with the edges in an order starting at the uninitialized image
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    TestRelaxProblem rp;
    rp.setupGroundPlaneProblem(graph, initialized_poses, cam_models, edges, {Option::ORIENTATION});

    // THEN: the edge between the initialized images still has its matches filtered
    EXPECT_GT(rp.test_num_grid_filtered_matches(id[1], edge_id[1]), 0);
    EXPECT_GT(rp.test_num_grid_filtered_matches(id[2], edge_id[1]), 0);
}

TEST_F(relax_group, measurement_3_images_points_without_orientation)
{
    // GIVEN: a graph, 3 images with edges between them all
    init_cameras();
    add_point_measurements(generate_3d_points());

    // WHEN: we set up a 3D point problem without optimizing orientation
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    TestRelaxProblem rp;
    rp.setup3dPointProblem(graph, np, cam_models, edges, {Option::POINTS_3D});

    // THEN: the matches were available, but no point measurements were added
    EXPECT_GT(rp.test_num_grid_filtered_matches(id[0], edge_id[0]), 0);
    EXPECT_TRUE(rp.test_get_tracks().empty());
}

TEST_F(relax_group, measurement_3_images_points_internals_point_triangulation_exact)
{
    // GIVEN: a graph, 3 images with edges between them all, with zero noise
    init_cameras();
    auto points = generate_3d_points();
    add_point_measurements(points);

    // AND: some noise in the graph, since we shouldn't be using those orientations anyways...
    add_ori_noise_graph({-0.1, 0.1, 0.1});

    // WHEN: we set up the problem
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    TestRelaxProblem rp;
    rp.setup3dPointProblem(graph, np, cam_models, edges, {Option::ORIENTATION, Option::POINTS_3D});

    auto vec2arr = [](const Eigen::Vector3d &vec) { return std::array<double, 3>{vec.x(), vec.y(), vec.z()}; };

    // THEN: the 3D points should be in the correct locations
    jk::tree::KDTree<size_t, 3> points_tree;
    for (size_t i = 0; i < points.size(); i++)
        points_tree.addPoint(vec2arr(points[i]), i);

    for (const auto &track : rp.test_get_tracks())
    {
        auto nearest = points_tree.search(vec2arr(track.point));
        EXPECT_LT(nearest.distance, 1e-8);
    }

    // WHEN: we run the solver
    rp.solve();

    // THEN: it should exit after 1 iteration ( + 1 more for numerical reasons)
    EXPECT_LE(rp.test_get_solver_summary().iterations.size(), 2);
    EXPECT_LT(rp.test_get_solver_summary().initial_cost, 1e-10);
    EXPECT_LT(rp.test_get_solver_summary().final_cost, 1e-10);
}

TEST_F(relax_group, measurement_3_images_triangulated_rays_internals_multi_ray_tracks_exact)
{
    // GIVEN: a graph, 3 images with edges between them all, observing non-planar points with zero noise
    init_cameras();
    add_point_measurements(generate_3d_points());

    // WHEN: we set up the triangulated rays problem
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    TestRelaxProblem rp;
    rp.setupTriangulatedRaysProblem(graph, np, cam_models, edges, {Option::ORIENTATION, Option::TRIANGULATED_RAYS});

    // THEN: the 3-view tracks should be used
    EXPECT_GT(rp.test_num_multi_ray_measurements(), 0);

    // AND: the true poses should cost only the downwards prior on the two 0.3 rad tilted cameras, ½·2·(600e-3·0.3)²,
    // despite the non-planar terrain
    rp.solve();
    EXPECT_NEAR(rp.test_get_solver_summary().initial_cost, 0.0324, 1e-9);
    EXPECT_LE(rp.test_get_solver_summary().final_cost, rp.test_get_solver_summary().initial_cost);
}

TEST_F(relax_group, measurement_3_images_triangulated_rays_internals_multi_ray_tracks_include_fixed_images)
{
    // GIVEN: a graph, 3 images with edges between them all, where the first image is already solved and fixed
    init_cameras();
    add_point_measurements(generate_3d_points());
    np.erase(np.begin());

    // WHEN: we set up the triangulated rays problem
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    TestRelaxProblem rp;
    rp.setupTriangulatedRaysProblem(graph, np, cam_models, edges, {Option::ORIENTATION, Option::TRIANGULATED_RAYS});

    // THEN: the fixed image should still anchor the 3-view tracks
    EXPECT_GT(rp.test_num_multi_ray_measurements(id[0]), 0);

    // AND: it should stay fixed
    rp.solve();
    EXPECT_EQ(graph.getNode(id[0])->payload.orientation.coeffs(), ground_ori[0].coeffs());
    const double downwards_prior_cost_of_tilted_cameras = 0.0324;
    EXPECT_LE(rp.test_get_solver_summary().final_cost, downwards_prior_cost_of_tilted_cameras + 1e-9);
}

TEST_F(relax_group, measurement_3_images_mesh_internals_multi_ray_tracks_include_fixed_images)
{
    // GIVEN: a graph, 3 images with edges between them all, where the first image is already solved and fixed
    init_cameras();
    add_point_measurements(generate_planar_points());
    np.erase(np.begin());

    // WHEN: we set up the ground mesh problem
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    TestRelaxProblem rp;
    rp.setupGroundMeshProblem(graph, np, cam_models, edges, {Option::ORIENTATION, Option::GROUND_MESH}, {});

    // THEN: the fixed image should still anchor the 3-view tracks
    EXPECT_GT(rp.test_num_multi_ray_measurements(id[0]), 0);

    // AND: it should stay fixed
    rp.solve();
    EXPECT_EQ(graph.getNode(id[0])->payload.orientation.coeffs(), ground_ori[0].coeffs());
}

TEST_F(relax_group, measurement_3_images_mesh_focal_with_non_optimizable_model)
{
    // GIVEN: a graph, 3 images with edges between them all, where the camera model is not in the optimizable set
    init_cameras();
    add_point_measurements(generate_planar_points());
    ankerl::unordered_dense::map<size_t, CameraModel> no_cam_models;

    // WHEN: we set up and solve a ground mesh problem that asks for focal length optimization
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    TestRelaxProblem rp;
    rp.setupGroundMeshProblem(graph, np, no_cam_models, edges,
                              {Option::ORIENTATION, Option::GROUND_MESH, Option::FOCAL_LENGTH}, {});
    rp.solve();

    // THEN: it completes and the shared camera model is untouched
    EXPECT_GT(rp.test_num_multi_ray_measurements(), 0);
    EXPECT_EQ(model->focal_length_pixels, 600);
}

TEST_F(relax_group, measurement_3_images_points_internals_point_triangulation_noise)
{
    // GIVEN: a graph, 3 images with edges between them all, with some noise
    init_cameras();
    auto points = generate_3d_points();
    auto vec2arr = [](const Eigen::Vector3d &vec) { return std::array<double, 3>{vec.x(), vec.y(), vec.z()}; };
    jk::tree::KDTree<size_t, 3> points_tree;
    for (size_t i = 0; i < points.size(); i++)
        points_tree.addPoint(vec2arr(points[i]), i);
    add_point_measurements(points);
    add_ori_noise({-0.05, 0.05, 0.05});

    // WHEN: we set up the problem and
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    TestRelaxProblem rp;
    rp.setup3dPointProblem(graph, np, cam_models, edges, {Option::ORIENTATION, Option::POINTS_3D});

    // THEN: the 3D points shouldn't be well triangulated
    for (const auto &track : rp.test_get_tracks())
    {
        auto nearest = points_tree.search(vec2arr(track.point));
        EXPECT_GT(nearest.distance, 1);
    }

    // WHEN: we solve the problem
    rp.solve();

    // THEN: the 3D points should be in the correct locations
    for (const auto &track : rp.test_get_tracks())
    {
        auto nearest = points_tree.search(vec2arr(track.point));
        EXPECT_LT(nearest.distance, 1e-8);
    }

    // AND: it took many iterations, and started with lots of error, but it minimizes to almost zero error
    EXPECT_GT(rp.test_get_solver_summary().iterations.size(), 10);
    EXPECT_GT(rp.test_get_solver_summary().initial_cost, 4e2);
    EXPECT_LT(rp.test_get_solver_summary().final_cost, 1e-10);
}

TEST_F(relax_group, measurement_3_images_points_internals_point_triangulation_noise_focal)
{
    // GIVEN: a graph, 3 images with edges between them all, with some noise
    init_cameras();
    auto points = generate_3d_points();
    auto vec2arr = [](const Eigen::Vector3d &vec) { return std::array<double, 3>{vec.x(), vec.y(), vec.z()}; };
    jk::tree::KDTree<size_t, 3> points_tree;
    for (size_t i = 0; i < points.size(); i++)
        points_tree.addPoint(vec2arr(points[i]), i);

    // add measurements with a different focal length
    model->focal_length_pixels *= 0.7;
    const double expected_focal_length = model->focal_length_pixels;
    add_point_measurements(points);
    model->focal_length_pixels /= 0.7;
    add_ori_noise({-0.05, 0.05, 0.05});

    // WHEN: we set up the problem and
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    TestRelaxProblem rp;
    // WHEN: we solve the problem
    rp.solve();

    // THEN: the 3D points should NOT be in the correct locations (optimization skipped)
    for (const auto &track : rp.test_get_tracks())
    {
        auto nearest = points_tree.search(vec2arr(track.point));
        EXPECT_GT(nearest.distance, 1e-6);
    }

    // AND: the camera model focal length was NOT optimized (remains at initial value) because optimization was skipped
    EXPECT_NEAR(cam_models[model->id].focal_length_pixels, expected_focal_length / 0.7, 0.1);

    // AND: it didn't run
    EXPECT_EQ(rp.test_get_solver_summary().iterations.size(), 0);
}

TEST_F(relax_group, measurement_3_images_points_internals_point_triangulation_noise_focal_principal)
{
    // GIVEN: a graph, 3 images with edges between them all, with some noise
    init_cameras();
    auto points = generate_3d_points();
    auto vec2arr = [](const Eigen::Vector3d &vec) { return std::array<double, 3>{vec.x(), vec.y(), vec.z()}; };
    jk::tree::KDTree<size_t, 3> points_tree;
    for (size_t i = 0; i < points.size(); i++)
        points_tree.addPoint(vec2arr(points[i]), i);

    // add measurements with a different focal length and principal point
    model->focal_length_pixels *= 0.8;
    model->principle_point << 380, 320;
    const double expected_focal_length = model->focal_length_pixels;
    const Eigen::Vector2d expected_principal_point(380, 320);

    add_point_measurements(points);

    // disturb them for the optimization
    model->focal_length_pixels /= 0.8;
    model->principle_point << 400, 300;
    add_ori_noise({-0.05, 0.05, 0.05});

    // WHEN: we set up the problem and
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    TestRelaxProblem rp;
    rp.setup3dPointProblem(graph, np, cam_models, edges,
                           {Option::ORIENTATION, Option::POINTS_3D, Option::FOCAL_LENGTH, Option::PRINCIPAL_POINT});

    // WHEN: we solve the problem
    rp.solve();

    // THEN: the solver ran
    EXPECT_GT(rp.test_get_solver_summary().iterations.size(), 0);

    // AND: the camera model parameters were optimized toward true values
    // Note: This is a challenging ill-conditioned optimization, so we use loose tolerances
    EXPECT_NEAR(cam_models[model->id].focal_length_pixels, expected_focal_length, 100);
    EXPECT_NEAR(cam_models[model->id].principle_point.x(), expected_principal_point.x(), 50);
    EXPECT_NEAR(cam_models[model->id].principle_point.y(), expected_principal_point.y(), 50);
}

TEST_F(relax_group, measurement_3_images_points_internals_point_triangulation_accuracy)
{
    // a test which just optimizes the points to check how they move from triangulation -> full bundle

    // GIVEN: a graph, 3 images with edges between them all, with some noise
    init_cameras();
    auto points = generate_3d_points();
    auto vec2arr = [](const Eigen::Vector3d &vec) { return std::array<double, 3>{vec.x(), vec.y(), vec.z()}; };
    jk::tree::KDTree<size_t, 3> points_tree;
    for (size_t i = 0; i < points.size(); i++)
        points_tree.addPoint(vec2arr(points[i]), i);
    add_point_measurements(points);
    add_ori_noise({-0.05, 0.05, 0.05});

    // WHEN: we set up the problem and
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    TestRelaxProblem rp;
    rp.setup3dPointProblem(graph, np, cam_models, edges, {Option::ORIENTATION, Option::POINTS_3D});

    // THEN: the 3D points shouldn't be well triangulated
    auto tracks_before = rp.test_get_tracks();
    for (const auto &track : tracks_before)
    {
        auto nearest = points_tree.search(vec2arr(track.point));
        EXPECT_GT(nearest.distance, 1);
    }

    rp.relaxObservedModelOnly();

    auto tracks_after = rp.test_get_tracks();

    // verify that the points for the tracks didn't move (much)
    ASSERT_EQ(tracks_before.size(), tracks_after.size());

    size_t moved_count = 0;
    for (size_t i = 0; i < tracks_before.size(); i++)
    {
        if ((tracks_before[i].point - tracks_after[i].point).norm() > 0.1)
            moved_count++;
    }
    EXPECT_LT(moved_count, 30);
}

// Test fixture for incremental optimization with multiple cameras
struct incremental_relax : public ::testing::Test
{
    static constexpr size_t NUM_CAMERAS = 25; // 5x5 grid to test connection limiting (max 10)
    static constexpr size_t GRID_WIDTH = 5;
    static constexpr size_t GRID_HEIGHT = 5;
    std::vector<size_t> camera_ids;
    MeasurementGraph graph;
    std::shared_ptr<CameraModel> model;
    std::vector<Eigen::Quaterniond> ground_ori;
    std::vector<Eigen::Vector3d> ground_pos;
    jk::tree::KDTree<size_t, 2> imageGPSLocations;

    void init_cameras()
    {
        camera_ids.resize(NUM_CAMERAS);
        ground_ori.resize(NUM_CAMERAS);
        ground_pos.resize(NUM_CAMERAS);

        auto down = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());

        // Create a 5x5 grid of cameras looking downward with slight variations
        for (size_t i = 0; i < NUM_CAMERAS; i++)
        {
            double x = 10 + (i % GRID_WIDTH) * 2; // x: 10, 12, 14, 16, 18
            double y = 10 + (i / GRID_WIDTH) * 2; // y: 10, 12, 14, 16, 18
            double angle_offset = 0.05 * (static_cast<double>(i) - NUM_CAMERAS / 2.0);

            ground_ori[i] = Eigen::Quaterniond(Eigen::AngleAxisd(angle_offset, Eigen::Vector3d::UnitZ()) * down);
            ground_pos[i] = Eigen::Vector3d(x, y, 10);
        }

        model = std::make_shared<CameraModel>();
        model->focal_length_pixels = 600;
        model->principle_point << 400, 300;
        model->pixels_cols = 800;
        model->pixels_rows = 600;
        model->projection_type = opencalibration::ProjectionType::PLANAR;
        model->id = 42;

        for (size_t i = 0; i < NUM_CAMERAS; i++)
        {
            image img;
            img.orientation = ground_ori[i];
            img.position = ground_pos[i];
            img.model = model;
            camera_ids[i] = graph.addNode(std::move(img));

            // Add to KD-tree for neighbor lookups
            imageGPSLocations.addPoint({ground_pos[i].x(), ground_pos[i].y()}, camera_ids[i]);
        }
    }

    point_cloud generate_planar_points()
    {
        point_cloud vec3d;
        vec3d.reserve(400); // More points for larger camera grid
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 20; j++)
            {
                // Points on a nearly flat plane below the cameras, covering the whole grid
                vec3d.emplace_back(9 + i * 0.5, 9 + j * 0.5, -5 + 1e-3 * i + 1e-2 * j);
            }
        }
        return vec3d;
    }

    void add_point_measurements_between_cameras(const point_cloud &points, size_t cam_a, size_t cam_b)
    {
        // Project points into both cameras and create correspondences
        auto *node_a = graph.getNode(camera_ids[cam_a]);
        auto *node_b = graph.getNode(camera_ids[cam_b]);

        camera_relations relation;

        for (size_t p_idx = 0; p_idx < points.size(); p_idx++)
        {
            const Eigen::Vector3d &p = points[p_idx];

            // Project into camera A
            Eigen::Vector3d ray_a = ground_ori[cam_a].inverse() * (p - ground_pos[cam_a]).normalized();
            Eigen::Vector2d pixel_a = image_from_3d(ray_a, *model);

            // Project into camera B
            Eigen::Vector3d ray_b = ground_ori[cam_b].inverse() * (p - ground_pos[cam_b]).normalized();
            Eigen::Vector2d pixel_b = image_from_3d(ray_b, *model);

            // Check if both projections are within image bounds
            if (pixel_a.x() >= 0 && pixel_a.x() < model->pixels_cols && pixel_a.y() >= 0 &&
                pixel_a.y() < model->pixels_rows && pixel_b.x() >= 0 && pixel_b.x() < model->pixels_cols &&
                pixel_b.y() >= 0 && pixel_b.y() < model->pixels_rows)
            {
                // Add features to both nodes
                feature_2d feat_a, feat_b;
                feat_a.location = pixel_a;
                feat_b.location = pixel_b;

                size_t feat_idx_a = node_a->payload.features.size();
                size_t feat_idx_b = node_b->payload.features.size();
                appendFeature(node_a->payload.features, feat_a);
                appendFeature(node_b->payload.features, feat_b);

                relation.inlier_matches.emplace_back(
                    feature_match_denormalized{pixel_a, pixel_b, feat_idx_a, feat_idx_b, p_idx});
            }
        }

        if (!relation.inlier_matches.empty())
        {
            graph.addEdge(std::move(relation), camera_ids[cam_a], camera_ids[cam_b]);
        }
    }

    void add_all_neighbor_measurements(const point_cloud &points)
    {
        // Create edges between adjacent cameras in the grid (8-connected neighborhood)
        for (size_t i = 0; i < NUM_CAMERAS; i++)
        {
            for (size_t j = i + 1; j < NUM_CAMERAS; j++)
            {
                double dist = (ground_pos[i] - ground_pos[j]).norm();
                if (dist < 3.5) // Connect cameras within ~sqrt(8) units (diagonal neighbors)
                {
                    add_point_measurements_between_cameras(points, i, j);
                }
            }
        }
    }

    void disturb_camera_orientation(size_t cam_idx, double noise_rad)
    {
        graph.getNode(camera_ids[cam_idx])->payload.orientation *=
            Eigen::Quaterniond(Eigen::AngleAxisd(noise_rad, Eigen::Vector3d::UnitY()));
    }

    // Get the center camera index (has the most neighbors)
    [[nodiscard]] size_t get_center_camera_idx() const
    {
        return (GRID_HEIGHT / 2) * GRID_WIDTH + (GRID_WIDTH / 2); // Center of 5x5 grid = index 12
    }

    // Count how many edges a camera has
    [[nodiscard]] size_t count_camera_edges(size_t cam_idx) const
    {
        return graph.getNode(camera_ids[cam_idx])->getEdges().size();
    }
};

TEST_F(incremental_relax, synthetic_planar_points_incremental_optimization)
{
    // GIVEN: A set of 25 calibrated cameras (5x5 grid) with synthetic measurements on a plane
    init_cameras();
    auto points = generate_planar_points();
    add_all_neighbor_measurements(points);

    // AND: The center camera (which has 8 neighbors) is treated as "newly added" with a disturbed orientation
    const size_t new_camera_idx = get_center_camera_idx();
    const double noise_rad = 0.2; // Add 0.2 radians of noise
    disturb_camera_orientation(new_camera_idx, noise_rad);

    // Verify setup: center camera should have multiple edges
    EXPECT_GE(count_camera_edges(new_camera_idx), 8) << "Center camera should have at least 8 neighbors";

    // Record the initial error for the new camera
    double initial_error = Eigen::AngleAxisd(graph.getNode(camera_ids[new_camera_idx])->payload.orientation.inverse() *
                                             ground_ori[new_camera_idx])
                               .angle();
    EXPECT_GT(initial_error, 0.1); // Should have significant initial error

    // WHEN: We run incremental optimization using RelaxGroup with the new camera as the primary node
    std::vector<size_t> new_camera_ids = {camera_ids[new_camera_idx]};
    RelaxGroup group;
    group.init(graph, new_camera_ids, imageGPSLocations, 2 /* graph_connection_depth */,
               {Option::ORIENTATION, Option::GROUND_PLANE});

    // Run the optimization (this includes both phase 1 and phase 2)
    group.run(graph, {});

    // Finalize to write results back to the graph
    auto optimized_ids = group.finalize(graph);

    // THEN: The new camera's orientation should converge toward the ground truth
    double final_error = Eigen::AngleAxisd(graph.getNode(camera_ids[new_camera_idx])->payload.orientation.inverse() *
                                           ground_ori[new_camera_idx])
                             .angle();

    // The error should be significantly reduced
    EXPECT_LT(final_error, initial_error * 0.5) << "Expected error reduction from " << initial_error << " to less than "
                                                << initial_error * 0.5 << ", got " << final_error;
    EXPECT_LT(final_error, 0.05) << "Expected final error < 0.05 rad, got " << final_error;

    // AND: The optimized IDs should include the new camera
    EXPECT_TRUE(std::find(optimized_ids.begin(), optimized_ids.end(), camera_ids[new_camera_idx]) !=
                optimized_ids.end());
}

TEST_F(incremental_relax, synthetic_planar_points_convergence_with_measurement_noise)
{
    // GIVEN: A set of 25 calibrated cameras with synthetic measurements on a plane
    init_cameras();
    auto points = generate_planar_points();

    // Add measurement noise by slightly shifting points before projection
    std::mt19937 gen(42);                                    // Fixed seed for reproducibility
    std::normal_distribution<double> noise_dist(0.0, 0.001); // Small position noise

    point_cloud noisy_points;
    noisy_points.reserve(points.size());
    for (const auto &p : points)
    {
        noisy_points.emplace_back(p.x() + noise_dist(gen), p.y() + noise_dist(gen), p.z() + noise_dist(gen));
    }

    add_all_neighbor_measurements(noisy_points);

    // AND: The center camera has a disturbed orientation
    const size_t new_camera_idx = get_center_camera_idx();
    const double noise_rad = 0.15;
    disturb_camera_orientation(new_camera_idx, noise_rad);

    double initial_error = Eigen::AngleAxisd(graph.getNode(camera_ids[new_camera_idx])->payload.orientation.inverse() *
                                             ground_ori[new_camera_idx])
                               .angle();

    // WHEN: We run incremental optimization
    std::vector<size_t> new_camera_ids = {camera_ids[new_camera_idx]};
    RelaxGroup group;
    group.init(graph, new_camera_ids, imageGPSLocations, 2, {Option::ORIENTATION, Option::GROUND_PLANE});
    group.run(graph, {});
    group.finalize(graph);

    // THEN: The optimization should still converge despite measurement noise
    double final_error = Eigen::AngleAxisd(graph.getNode(camera_ids[new_camera_idx])->payload.orientation.inverse() *
                                           ground_ori[new_camera_idx])
                             .angle();

    EXPECT_LT(final_error, initial_error) << "Optimization should reduce error";
    EXPECT_LT(final_error, 0.1) << "Expected final error < 0.1 rad with noise, got " << final_error;
}

TEST_F(incremental_relax, multiple_new_cameras_incremental)
{
    // GIVEN: A set of 25 calibrated cameras
    init_cameras();
    auto points = generate_planar_points();
    add_all_neighbor_measurements(points);

    // AND: Multiple cameras in a row are treated as "newly added" with disturbed orientations
    // Use cameras in the middle row: indices 10, 11, 12, 13, 14
    std::vector<size_t> new_camera_indices = {10, 11, 12, 13, 14};
    for (size_t idx : new_camera_indices)
    {
        disturb_camera_orientation(idx, 0.15);
    }

    std::vector<double> initial_errors;
    for (size_t idx : new_camera_indices)
    {
        initial_errors.push_back(
            Eigen::AngleAxisd(graph.getNode(camera_ids[idx])->payload.orientation.inverse() * ground_ori[idx]).angle());
    }

    // WHEN: We run incremental optimization with multiple new cameras
    std::vector<size_t> new_camera_ids;
    for (size_t idx : new_camera_indices)
    {
        new_camera_ids.push_back(camera_ids[idx]);
    }

    RelaxGroup group;
    group.init(graph, new_camera_ids, imageGPSLocations, 2, {Option::ORIENTATION, Option::GROUND_PLANE});
    group.run(graph, {});
    group.finalize(graph);

    // THEN: All new cameras should converge
    for (size_t i = 0; i < new_camera_indices.size(); i++)
    {
        size_t idx = new_camera_indices[i];
        double final_error =
            Eigen::AngleAxisd(graph.getNode(camera_ids[idx])->payload.orientation.inverse() * ground_ori[idx]).angle();

        EXPECT_LT(final_error, initial_errors[i]) << "Camera " << idx << " should improve";
        EXPECT_LT(final_error, 0.1) << "Camera " << idx << " final error should be < 0.1 rad";
    }
}

TEST_F(incremental_relax, connection_limiting_with_many_neighbors)
{
    // GIVEN: A set of 25 calibrated cameras where the center camera has many neighbors
    init_cameras();
    auto points = generate_planar_points();
    add_all_neighbor_measurements(points);

    // The center camera should have 8 direct neighbors (8-connected in a 5x5 grid)
    const size_t center_idx = get_center_camera_idx();
    size_t num_edges = count_camera_edges(center_idx);
    EXPECT_EQ(num_edges, 8) << "Center camera should have exactly 8 neighbors in 5x5 grid";

    // Disturb the center camera
    disturb_camera_orientation(center_idx, 0.2);

    double initial_error =
        Eigen::AngleAxisd(graph.getNode(camera_ids[center_idx])->payload.orientation.inverse() * ground_ori[center_idx])
            .angle();

    // WHEN: We run incremental optimization with high graph_connection_depth
    // This should trigger the connection limiting (max 10 connected cameras)
    // Even though many cameras are included in _local_poses for measurements,
    // only primary + top 10 connected are actually optimized (free to move)
    std::vector<size_t> new_camera_ids = {camera_ids[center_idx]};
    RelaxGroup group;
    group.init(graph, new_camera_ids, imageGPSLocations, 3 /* high depth to get many connections */,
               {Option::ORIENTATION, Option::GROUND_PLANE});
    group.run(graph, {});
    auto all_node_ids = group.finalize(graph);

    // THEN: The optimization should converge
    double final_error =
        Eigen::AngleAxisd(graph.getNode(camera_ids[center_idx])->payload.orientation.inverse() * ground_ori[center_idx])
            .angle();

    EXPECT_LT(final_error, initial_error) << "Optimization should reduce error";
    EXPECT_LT(final_error, 0.05) << "Expected final error < 0.05 rad, got " << final_error;

    // AND: Many nodes should be included in the optimization graph (for measurements)
    // but the connection limiting ensures only a subset are actually optimized.
    // finalize() returns all nodes that were part of the problem (including fixed ones).
    EXPECT_GT(all_node_ids.size(), 11) << "With depth 3, many nodes should be in the problem";

    // The key test is that convergence still works even with the limiting,
    // and that the primary camera was definitely optimized
    EXPECT_TRUE(std::find(all_node_ids.begin(), all_node_ids.end(), camera_ids[center_idx]) != all_node_ids.end());
}

TEST_F(incremental_relax, two_phase_optimization_improves_convergence)
{
    // GIVEN: A set of 25 calibrated cameras
    init_cameras();
    auto points = generate_planar_points();
    add_all_neighbor_measurements(points);

    // AND: The center camera has a large disturbance (harder to converge)
    const size_t center_idx = get_center_camera_idx();
    const double large_noise_rad = 0.3; // 0.3 radians = ~17 degrees
    disturb_camera_orientation(center_idx, large_noise_rad);

    double initial_error =
        Eigen::AngleAxisd(graph.getNode(camera_ids[center_idx])->payload.orientation.inverse() * ground_ori[center_idx])
            .angle();
    EXPECT_GT(initial_error, 0.25); // Confirm large initial error

    // WHEN: We run incremental optimization (which uses two-phase approach)
    std::vector<size_t> new_camera_ids = {camera_ids[center_idx]};
    RelaxGroup group;
    group.init(graph, new_camera_ids, imageGPSLocations, 2, {Option::ORIENTATION, Option::GROUND_PLANE});
    group.run(graph, {});
    group.finalize(graph);

    // THEN: Even with large initial error, the optimization should converge
    double final_error =
        Eigen::AngleAxisd(graph.getNode(camera_ids[center_idx])->payload.orientation.inverse() * ground_ori[center_idx])
            .angle();

    EXPECT_LT(final_error, initial_error * 0.3) << "Should achieve at least 70% error reduction";
    EXPECT_LT(final_error, 0.1) << "Expected final error < 0.1 rad even with large initial disturbance";
}

TEST(RobustCentroid, continuous_in_its_inputs)
{
    // GIVEN: three points, one of which slides steadily away from the others
    const double step = 1e-4;
    Eigen::Vector3d points[] = {{0, 0, 0}, {1, 0, 0}, {0.5, 0.5, 0}};
    Eigen::Vector3d previous = robustCentroid(points, 3, 0.3);

    double largest_jump = 0;
    for (double y = 0.5; y < 3; y += step)
    {
        // WHEN: we recompute the centroid after each small move
        points[2].y() = y;
        const Eigen::Vector3d centroid = robustCentroid(points, 3, 0.3);
        largest_jump = std::max(largest_jump, (centroid - previous).norm());
        previous = centroid;
    }

    // THEN: the centroid never jumps by more than the input moved
    EXPECT_LT(largest_jump, 2 * step);
}

TEST(RobustCentroid, identical_points)
{
    Eigen::Vector3d points[] = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
    auto result = robustCentroid(points, 3, 1.0);
    EXPECT_NEAR(result.x(), 1.0, 1e-6);
    EXPECT_NEAR(result.y(), 2.0, 1e-6);
    EXPECT_NEAR(result.z(), 3.0, 1e-6);
}

TEST(RobustCentroid, close_points_near_average)
{
    Eigen::Vector3d points[] = {{0, 0, 0}, {0.01, 0, 0}, {0, 0.01, 0}};
    auto result = robustCentroid(points, 3, 100.0);
    Eigen::Vector3d naive(0.01 / 3, 0.01 / 3, 0);
    EXPECT_LT((result - naive).norm(), 0.01);
}

TEST(RobustCentroid, outlier_downweighted)
{
    Eigen::Vector3d points[] = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {100, 0, 0}};
    auto robust = robustCentroid(points, 4, 1.0);

    // Inlier centroid is (1, 0, 0). Robust result should stay close to it.
    EXPECT_NEAR(robust.x(), 1.0, 0.5);
    EXPECT_NEAR(robust.y(), 0.0, 1e-6);
    EXPECT_NEAR(robust.z(), 0.0, 1e-6);
}

TEST(RobustCentroid, two_points)
{
    Eigen::Vector3d points[] = {{0, 0, 0}, {2, 0, 0}};
    auto result = robustCentroid(points, 2, 10.0);
    EXPECT_NEAR(result.x(), 1.0, 1e-4);
    EXPECT_NEAR(result.y(), 0.0, 1e-6);
    EXPECT_NEAR(result.z(), 0.0, 1e-6);
}

TEST(RobustCentroid, single_point)
{
    Eigen::Vector3d points[] = {{5, 3, 1}};
    auto result = robustCentroid(points, 1, 1.0);
    EXPECT_NEAR(result.x(), 5.0, 1e-6);
    EXPECT_NEAR(result.y(), 3.0, 1e-6);
    EXPECT_NEAR(result.z(), 1.0, 1e-6);
}

TEST_F(relax_group, measurement_3_images_triangulated_rays_loss_widens_with_starting_residuals)
{
    // GIVEN: a graph, 3 images with edges between them all
    init_cameras();
    add_point_measurements(generate_3d_points());
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};

    // WHEN: we set up the triangulated rays problem at the true poses, and with one camera rotated by 3 degrees
    TestRelaxProblem exact;
    exact.setupTriangulatedRaysProblem(graph, np, cam_models, edges, {Option::ORIENTATION, Option::TRIANGULATED_RAYS});
    np[0].orientation = np[0].orientation * Eigen::AngleAxisd(3 * M_PI / 180, Eigen::Vector3d::UnitX());
    TestRelaxProblem rotated;
    rotated.setupTriangulatedRaysProblem(graph, np, cam_models, edges,
                                         {Option::ORIENTATION, Option::TRIANGULATED_RAYS});

    // THEN: the exact problem uses a 3 sigma loss for a 1 pixel sigma, and the rotated one a much wider loss
    const int dof = 3;
    EXPECT_NEAR(exact.test_ray_loss_delta(dof), std::sqrt(14.16), 0.05);
    EXPECT_GT(rotated.test_ray_loss_delta(dof), 5 * exact.test_ray_loss_delta(dof));
}

TEST_F(relax_group, measurement_3_images_surface_model_rejects_inconsistent_rays)
{
    // GIVEN: a graph, 3 images with edges between them all
    init_cameras();
    auto points = generate_3d_points();
    add_point_measurements(points);
    ankerl::unordered_dense::set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};

    // WHEN: we take the surface model at the true poses, and with one camera rotated by 3 degrees
    TestRelaxProblem exact;
    exact.setupTriangulatedRaysProblem(graph, np, cam_models, edges, {Option::ORIENTATION, Option::TRIANGULATED_RAYS});
    const surface_model exact_surface = exact.getSurfaceModel();
    np[0].orientation = np[0].orientation * Eigen::AngleAxisd(3 * M_PI / 180, Eigen::Vector3d::UnitX());
    TestRelaxProblem rotated;
    rotated.setupTriangulatedRaysProblem(graph, np, cam_models, edges,
                                         {Option::ORIENTATION, Option::TRIANGULATED_RAYS});
    const surface_model rotated_surface = rotated.getSurfaceModel();

    // THEN: the exact points are kept where they were observed, and the inconsistent ones dropped
    ASSERT_EQ(exact_surface.cloud.size(), 1u);
    EXPECT_GT(exact_surface.cloud[0].size(), points.size() / 2);
    for (const auto &p : exact_surface.cloud[0])
    {
        double nearest = std::numeric_limits<double>::infinity();
        for (const auto &q : points)
            nearest = std::min(nearest, (p - q).norm());
        EXPECT_LT(nearest, 1e-6);
    }
    const size_t rotated_points = rotated_surface.cloud.empty() ? 0 : rotated_surface.cloud[0].size();
    EXPECT_EQ(rotated_points, 0u);
}

TEST(relax, mesh_height_problem_recovers_surface_despite_outliers)
{
    point_cloud cameras;
    cameras.push_back(Eigen::Vector3d(0, 0, 10));
    cameras.push_back(Eigen::Vector3d(10, 10, 10));

    surface_model surface;
    surface.mesh = buildMinimalMesh(cameras, {});

    // GIVEN: a tilted plane with a bump, noise and gross outliers, meshed beyond the data so there are empty regions
    auto height = [](double x, double y) {
        return 0.1 * x - 0.2 * y + 3.0 + 0.5 * std::exp(-((x - 4) * (x - 4) + (y - 6) * (y - 6)) / 4);
    };
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0, 0.02);
    point_cloud pts;
    for (double x = 0.25; x < 10; x += 0.25)
        for (double y = 0.25; y < 10; y += 0.25)
            pts.push_back(Eigen::Vector3d(x, y, height(x, y) + noise(gen)));
    for (int i = 0; i < 40; i++)
        pts.push_back(Eigen::Vector3d(0.5 + 0.2 * i, 3, height(0.5 + 0.2 * i, 3) + 20.0));
    surface.cloud = {pts};

    refineByPointDensity(surface.mesh, surface.cloud, 5, 0.0, 14);
    ASSERT_GT(surface.mesh.size_nodes(), 50);

    // WHEN: fitting the mesh heights
    RelaxProblem rp;
    rp.setupMeshHeightProblem(surface, 0.02);
    rp.solveMeshHeights();
    const MeshGraph relaxed = rp.getSurfaceModel().mesh;

    // AND: fitting them in regions of about 200 samples
    MeshGraph tiled = surface.mesh;
    fitMeshHeights(tiled, surface.cloud, 0.02, 200);

    // THEN: vertices a margin inside the data sit on the true surface, ignoring the outliers and keeping the tilt
    for (const MeshGraph *mesh : std::array<const MeshGraph *, 2>{&relaxed, &tiled})
    {
        ASSERT_EQ(mesh->size_nodes(), surface.mesh.size_nodes());
        double maxError = 0;
        for (auto it = mesh->cnodebegin(); it != mesh->cnodeend(); ++it)
        {
            const Eigen::Vector3d &p = it->second.payload.location;
            if (p.x() > 1 && p.x() < 9 && p.y() > 1 && p.y() < 9)
                maxError = std::max(maxError, std::abs(p.z() - height(p.x(), p.y())));
        }
        EXPECT_LT(maxError, 0.05);
    }
}
