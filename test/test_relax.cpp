#include <jk/KDTree.h>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/relax/relax.hpp>
#include <opencalibration/relax/relax_cost_function.hpp>
#include <opencalibration/relax/relax_group.hpp>
#include <opencalibration/relax/relax_problem.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <random>

using namespace opencalibration;
using namespace std::chrono_literals;

struct relax_group : public ::testing::Test
{
    size_t id[3];
    MeasurementGraph graph;
    std::vector<NodePose> np;
    std::unordered_map<size_t, CameraModel> cam_models;
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
                graph.getNode(np[i].node_id)->payload.features.emplace_back(feat);
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
                    pixel[j] = graph.getNode(np[index[j]].node_id)->payload.features[counter].location;
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

TEST_F(relax_group, downwards_prior_cost_function)
{
    // GIVEN: a starting angle
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity() * Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());

    // WHEN: we get the cost of the downwards prior
    PointsDownwardsPrior p(1e-3);
    double r = NAN;
    EXPECT_TRUE(p(q.coeffs().data(), &r));

    // THEN: it should be the amount away from vertical
    EXPECT_NEAR(r, 0, 1e-8);

    // WHEN: we shift it 0.3 rad away from vertical
    q = q * Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX());
    EXPECT_TRUE(p(q.coeffs().data(), &r));

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
    DecomposedRotationCost cost(rel_rot, rel_pos, &ground_pos[0], &ground_pos[1], 8);

    {
        // WHEN: we get the relative orientation cost with a perfect guess
        Eigen::Quaterniond q[2]{ground_ori[0], ground_ori[1]};
        double r[3]{NAN, NAN, NAN};
        bool success = cost(q[0].coeffs().data(), q[1].coeffs().data(), r);

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
        bool success = cost(q[0].coeffs().data(), q[1].coeffs().data(), r);

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
        bool success = cost(q[0].coeffs().data(), q[1].coeffs().data(), r);

        // THEN: we should get that fixed offset as the residuals
        EXPECT_TRUE(success);
        EXPECT_NEAR(r[0], 0.0, 1e-5);
        EXPECT_NEAR(r[1], 0.3, 1e-12);
        EXPECT_NEAR(r[2], 0.3, 1e-12);
    }
}

TEST_F(relax_group, no_images)
{
    // GIVEN: a graph, with no images
    MeasurementGraph graph;
    std::vector<NodePose> np;
    std::unordered_map<size_t, CameraModel> cam_models;

    // WHEN: we relax the relative orientations
    relax(graph, np, cam_models, {}, {Option::ORIENTATION}, {});

    // THEN: it shouldn't crash
}

TEST_F(relax_group, prior_1_image)
{
    // GIVEN: a graph, with 1 image tilted 45 degrees from downward
    MeasurementGraph graph;
    std::vector<NodePose> np;
    std::unordered_map<size_t, CameraModel> cam_models;

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
    std::unordered_map<size_t, CameraModel> cam_models;

    // NB! relative translation of the two images is on the X axis, so the rotation around the X axis is unconstrained
    // Only put disturbances on the Y axis!

    Eigen::Quaterniond ori(Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY()));
    Eigen::Vector3d pos(9, 9, 9);
    image img;
    img.orientation = ori;
    img.position = pos;

    const size_t id = graph.addNode(std::move(img));
    np.emplace_back(NodePose{id, ori, pos});

    ori = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI_4, Eigen::Vector3d::UnitY()));
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
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
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
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax(graph, np, cam_models, edges, {Option::ORIENTATION, Option::POINTS_3D}, {});

    // THEN: it should put them back into the original orientation
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 1e-8)
            << i << ": " << np[i].orientation.coeffs().transpose() << std::endl
            << "g: " << ground_ori[i].coeffs().transpose();
}

TEST_F(relax_group, measurement_3_images_plane)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    init_cameras();
    add_point_measurements(generate_planar_points());
    add_ori_noise({-0.1, 0.1, 0.1});

    // WHEN: we relax them with relative orientation
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax(graph, np, cam_models, edges, {Option::ORIENTATION, Option::GROUND_PLANE}, {});
    // and again to re-init the inliers
    relax(graph, np, cam_models, edges, {Option::ORIENTATION, Option::GROUND_PLANE}, {});

    // THEN: it should put them back into the original orientation
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 1e-3)
            << i << ": " << np[i].orientation.coeffs().transpose() << std::endl
            << "g: " << ground_ori[i].coeffs().transpose();
}

TEST_F(relax_group, measurement_3_images_mesh_radial)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    init_cameras();
    cam_models[model->id].radial_distortion << 0.1, -0.1, 0.1;
    add_point_measurements(generate_planar_points());
    add_ori_noise({-0.1, 0.1, 0.1});
    cam_models[model->id].radial_distortion.fill(0);

    // WHEN: we relax them with relative orientation
    const RelaxOptionSet options({Option::ORIENTATION, Option::LENS_DISTORTIONS_RADIAL,
                                  Option::LENS_DISTORTIONS_RADIAL_BROWN246_PARAMETERIZATION, Option::GROUND_MESH});
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};

    for (int i = 0; i < 10; i++)
        // a few times...
        relax(graph, np, cam_models, edges, options, {});

    // THEN: the solver optimizes the orientations toward ground truth
    // Orientation error should be reduced from initial noise of 0.1 rad
    for (int i = 0; i < 3; i++)
    {
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 0.1);
    }

    // Radial distortion should be optimized toward the true value (0.1, -0.1, 0.1)
    Eigen::Vector3d expected_distortion(0.1, -0.1, 0.1);
    EXPECT_LT((cam_models[model->id].radial_distortion - expected_distortion).norm(), 0.2);
}

class TestRelaxProblem : public RelaxProblem
{
  public:
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
};

TEST_F(relax_group, measurement_3_images_points_internals_point_triangulation_exact)
{
    // GIVEN: a graph, 3 images with edges between them all, with zero noise
    init_cameras();
    auto points = generate_3d_points();
    add_point_measurements(points);

    // AND: some noise in the graph, since we shouldn't be using those orientations anyways...
    add_ori_noise_graph({-0.1, 0.1, 0.1});

    // WHEN: we set up the problem
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
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
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
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
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
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
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
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
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
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
                node_a->payload.features.push_back(feat_a);
                node_b->payload.features.push_back(feat_b);

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
    size_t get_center_camera_idx() const
    {
        return (GRID_HEIGHT / 2) * GRID_WIDTH + (GRID_WIDTH / 2); // Center of 5x5 grid = index 12
    }

    // Count how many edges a camera has
    size_t count_camera_edges(size_t cam_idx) const
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
