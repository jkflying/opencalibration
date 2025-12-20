#include <jk/KDTree.h>
#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/relax/relax.hpp>
#include <opencalibration/relax/relax_cost_function.hpp>
#include <opencalibration/relax/relax_problem.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <gtest/gtest.h>

#include <chrono>

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
    // GIVEN: a graph, with 1 image
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

    // WHEN: we relax the relative orientations
    relax(graph, np, cam_models, {}, {Option::ORIENTATION}, {});

    // THEN: it should be skipped due to rank deficiency
    EXPECT_NEAR((np[0].orientation.coeffs() - ori.coeffs()).norm(), 0.0, 1e-9);
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

    // THEN: it should be skipped due to rank deficiency
    EXPECT_NEAR((np[0].orientation.coeffs() - img.orientation.coeffs()).norm(), 0.0, 1e-9);
    EXPECT_NEAR((np[1].orientation.coeffs() - img2.orientation.coeffs()).norm(), 0.0, 1e-9);
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

    // THEN: it should be skipped due to rank deficiency, so parameters remain at initial (noisy) values
    // Orientation error should be significant (initial noise was 0.1 rad)
    for (int i = 0; i < 3; i++)
    {
        EXPECT_GT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 1e-2);
    }

    // Radial distortion should remain 0 (initial) instead of optimizing to 0.1
    EXPECT_NEAR(cam_models[model->id].radial_distortion.norm(), 0.0, 1e-9);
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

    // THEN: the 3D points should NOT be in the correct locations (optimization skipped)
    for (const auto &track : rp.test_get_tracks())
    {
        auto nearest = points_tree.search(vec2arr(track.point));
        EXPECT_GT(nearest.distance, 1e-6);
    }

    // AND: the camera model parameters were NOT optimized (remain at initial values) because optimization was skipped
    EXPECT_NEAR(cam_models[model->id].focal_length_pixels, expected_focal_length / 0.8, 0.1);
    EXPECT_NEAR(cam_models[model->id].principle_point.x(), 400, 0.1);
    EXPECT_NEAR(cam_models[model->id].principle_point.y(), 300, 0.1);

    // AND: it didn't run
    EXPECT_EQ(rp.test_get_solver_summary().iterations.size(), 0);
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
