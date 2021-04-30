#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/relax/relax.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>
#include <opencalibration/types/point_cloud.hpp>

#include <gtest/gtest.h>

#include <chrono>

using namespace opencalibration;
using namespace std::chrono_literals;

struct relax_ : public ::testing::Test
{
    size_t id[3];
    MeasurementGraph graph;
    std::vector<NodePose> np;
    CameraModel model;
    Eigen::Quaterniond ground_ori[3];
    Eigen::Vector3d ground_pos[3];
    size_t edge_id[3];

    void init_cameras()
    {
        ground_ori[0] = Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()));
        ground_ori[1] = Eigen::Quaterniond(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitY()));
        ground_ori[2] = Eigen::Quaterniond(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitX()));
        ground_pos[0] = Eigen::Vector3d(9, 9, 9);
        ground_pos[1] = Eigen::Vector3d(11, 9, 9);
        ground_pos[2] = Eigen::Vector3d(11, 11, 9);

        model.focal_length_pixels = 600;
        model.principle_point << 400, 300;
        model.pixels_cols = 800;
        model.pixels_rows = 600;
        model.projection_type = opencalibration::ProjectionType::PLANAR;

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
            camera_relations relation;
            size_t index[2] = {i, (i + 1) % 3};
            for (const Eigen::Vector3d &p : points)
            {

                Eigen::Vector2d pixel[2];
                for (int j = 0; j < 2; j++)
                {
                    Eigen::Vector3d ray = np[index[j]].orientation.inverse() * (np[index[j]].position - p).normalized();
                    pixel[j] = image_from_3d(ray, model);
                }
                relation.inlier_matches.emplace_back(feature_match_denormalized{pixel[0], pixel[1]});
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

            Eigen::Quaterniond actual_r = np[index[0]].orientation * np[index[1]].orientation.inverse();
            Eigen::Vector3d actual_t = actual_r * (np[index[1]].position - np[index[0]].position).normalized();

            if (i == 0 || i == 2)
            {
                relation.relative_translation = actual_t;
                relation.relative_rotation = actual_r;
            }
            if (i == 1 || i == 2)
            {
                relation.relative_translation2 = actual_t;
                relation.relative_rotation2 = actual_r;
            }

            edge_id[i] = graph.addEdge(std::move(relation), id[index[0]], id[index[1]]);
        }
    }

    void add_ori_noise(std::array<double, 3> noise)
    {
        np[0].orientation *= Eigen::Quaterniond(Eigen::AngleAxisd(noise[0], Eigen::Vector3d::UnitY()));
        np[1].orientation *= Eigen::Quaterniond(Eigen::AngleAxisd(noise[1], Eigen::Vector3d::UnitZ()));
        np[2].orientation *= Eigen::Quaterniond(Eigen::AngleAxisd(noise[2], Eigen::Vector3d::UnitX()));
    }
};

TEST(relax, no_images)
{
    // GIVEN: a graph, with no images
    MeasurementGraph graph;
    std::vector<NodePose> np;

    // WHEN: we relax the relative orientations
    relaxDecompositions(graph, np, {});

    // THEN: it shouldn't crash
}

TEST(relax, prior_1_image)
{
    // GIVEN: a graph, with 1 image
    MeasurementGraph graph;
    std::vector<NodePose> np;

    Eigen::Quaterniond ori(Eigen::AngleAxisd(M_PI_4, Eigen::Vector3d::UnitX()));
    Eigen::Vector3d pos(9, 9, 9);
    image img;
    img.orientation = ori;
    img.position = pos;

    size_t id = graph.addNode(std::move(img));
    np.emplace_back(NodePose{id, ori, pos});

    // WHEN: we relax the relative orientations
    relaxDecompositions(graph, np, {});

    // THEN: it should be pointing downwards, with other axes close to the original
    EXPECT_LT((np[0].orientation.coeffs() - Eigen::Vector4d(1, 0, 0, 0)).norm(), 1e-3)
        << np[0].orientation.coeffs().transpose();
}

TEST(relax, prior_2_images)
{
    // GIVEN: a graph, with 2 images with relative orientation as identity
    MeasurementGraph graph;
    std::vector<NodePose> np;

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
    relation.relative_rotation2 = Eigen::Quaterniond::Identity();
    relation.relative_translation2 << 1, 0, 0;
    size_t edge_id = graph.addEdge(std::move(relation), id, id2);

    // WHEN: we relax the relative orientations
    relaxDecompositions(graph, np, {edge_id});

    // THEN: it should be pointing downwards, with other axes close to the original
    EXPECT_LT(Eigen::AngleAxisd(np[0].orientation).angle(), 1e-5) << np[0].orientation.coeffs().transpose();
    EXPECT_LT(Eigen::AngleAxisd(np[1].orientation).angle(), 1e-5) << np[1].orientation.coeffs().transpose();
}

TEST_F(relax_, relative_orientation_3_images)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    init_cameras();

    // TODO: make it work with images that aren't pointed down to start with
    for (int i = 0; i < 3; i++)
    {
        np[i].orientation = Eigen::Quaterniond::Identity();
        ground_ori[i] = Eigen::Quaterniond::Identity();
    }

    add_edge_measurements();
    add_ori_noise({-1, 1, 1});

    // WHEN: we relax them with relative orientation
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relaxDecompositions(graph, np, edges);

    // THEN: it should put them back into the original orientation
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation * ground_ori[i].inverse()).angle(), 1e-5)
            << np[i].orientation.coeffs().transpose();
}

TEST_F(relax_, measurement_3_images_points)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    init_cameras();
    add_point_measurements(generate_3d_points());
    add_ori_noise({-0.05, 0.05, 0.05});

    // WHEN: we relax them with relative orientation
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax3dPointMeasurements(graph, np, edges);

    // THEN: it should put them back into the original orientation
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 1e-8)
            << i << ": " << np[i].orientation.coeffs().transpose() << std::endl
            << "g: " << ground_ori[i].coeffs().transpose();
}

TEST_F(relax_, measurement_3_images_plane)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    init_cameras();
    add_point_measurements(generate_planar_points());
    add_ori_noise({-0.3, 0.2, 0.2});

    // WHEN: we relax them with relative orientation
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relaxGroundPlaneMeasurements(graph, np, edges);

    // THEN: it should put them back into the original orientation
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 1e-9)
            << i << ": " << np[i].orientation.coeffs().transpose() << std::endl
            << "g: " << ground_ori[i].coeffs().transpose();
}
