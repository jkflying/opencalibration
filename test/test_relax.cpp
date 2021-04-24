#include <opencalibration/distort/distort_keypoints.hpp>
#include <opencalibration/relax/relax.hpp>
#include <opencalibration/types/measurement_graph.hpp>
#include <opencalibration/types/node_pose.hpp>

#include <gtest/gtest.h>

#include <chrono>

using namespace opencalibration;
using namespace std::chrono_literals;

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

TEST(relax, relative_orientation_3_images)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    MeasurementGraph graph;
    std::vector<NodePose> np;

    // NB! relative translation of the two images is on the X axis, so the rotation around the X axis is unconstrained
    // Only put disturbances on the Y axis!

    Eigen::Quaterniond ori(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
    Eigen::Vector3d pos(9, 9, 9);
    image img;
    img.orientation = ori;
    img.position = pos;

    const size_t id = graph.addNode(std::move(img));
    np.emplace_back(NodePose{id, ori, pos});

    ori = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI, Eigen::Vector3d::UnitY()));
    pos << 11, 9, 9;

    image img2;
    img2.orientation = ori;
    img2.position = pos;
    const size_t id2 = graph.addNode(std::move(img2));
    np.emplace_back(NodePose{id2, ori, pos});

    camera_relations relation;
    relation.relative_rotation = Eigen::Quaterniond::Identity();
    relation.relative_translation << 1, 0, 0;
    // leave secondary relation as NaN

    size_t edge_id = graph.addEdge(std::move(relation), id, id2);

    ori = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI, Eigen::Vector3d::UnitX()));
    pos << 11, 11, 9;

    image img3;
    img3.orientation = ori;
    img3.position = pos;
    const size_t id3 = graph.addNode(std::move(img3));
    np.emplace_back(NodePose{id3, ori, pos});

    // primary relation is garbage, second is correct
    camera_relations relation2;
    relation2.relative_rotation = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI_4, Eigen::Vector3d::UnitX()));
    relation2.relative_translation << 0, 0, 1;
    relation2.relative_rotation2 = Eigen::Quaterniond::Identity();
    relation2.relative_translation2 << 0, 1, 0;
    size_t edge_id2 = graph.addEdge(std::move(relation2), id2, id3);

    // primary relation is correct, second is garbage
    camera_relations relation3;
    relation3.relative_rotation = Eigen::Quaterniond::Identity();
    relation3.relative_translation = Eigen::Vector3d(1, 1, 0).normalized();
    relation3.relative_rotation2 = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI_4, Eigen::Vector3d::UnitZ()));
    relation3.relative_translation2 = Eigen::Vector3d(0, 1, 1).normalized();
    size_t edge_id3 = graph.addEdge(std::move(relation3), id, id3);

    // WHEN: we relax them with relative orientation
    relaxDecompositions(graph, np, {edge_id, edge_id2, edge_id3});

    // THEN: it should put them back into the original orientation
    EXPECT_LT(Eigen::AngleAxisd(np[0].orientation).angle(), 1e-5) << np[0].orientation.coeffs().transpose();
    EXPECT_LT(Eigen::AngleAxisd(np[1].orientation).angle(), 1e-5) << np[1].orientation.coeffs().transpose();
    EXPECT_LT(Eigen::AngleAxisd(np[2].orientation).angle(), 1e-5) << np[2].orientation.coeffs().transpose();
}

TEST(relax, measurement_3_images_points)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    MeasurementGraph graph;
    std::vector<NodePose> np;
    CameraModel model;
    model.focal_length_pixels = 600;
    model.principle_point << 400, 300;
    model.pixels_cols = 800;
    model.pixels_rows = 600;
    model.projection_type = opencalibration::ProjectionType::PLANAR;

    size_t id[3];
    Eigen::Quaterniond ori0(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond ori1(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitY()));
    Eigen::Quaterniond ori2(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitX()));
    Eigen::Quaterniond ground_ori[3]{ori0, ori1, ori2};
    Eigen::Vector3d pos0(9, 9, 9);
    Eigen::Vector3d pos1(11, 9, 9);
    Eigen::Vector3d pos2(11, 11, 9);
    Eigen::Vector3d ground_pos[3]{pos0, pos1, pos2};

    for (int i = 0; i < 3; i++)
    {
        image img;
        img.orientation = ground_ori[i];
        img.position = ground_pos[i];
        img.model = model;
        id[i] = graph.addNode(std::move(img));
        np.emplace_back(NodePose{id[i], ground_ori[i], ground_pos[i]});
    }
    // 3d points that they all see

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vec3d;
    vec3d.reserve(100);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            vec3d.emplace_back(i + 5, j + 5, -10 + (i + j) % 2);
        }
    }

    size_t edge_id[3];
    for (size_t i = 0; i < 3; i++)
    {
        camera_relations relation;
        size_t index[2] = {i, (i + 1) % 3};
        for (const Eigen::Vector3d &p : vec3d)
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

    // add some noise to the orientations
    np[0].orientation *= Eigen::Quaterniond(Eigen::AngleAxisd(-0.05, Eigen::Vector3d::UnitY()));
    np[1].orientation *= Eigen::Quaterniond(Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ()));
    np[2].orientation *= Eigen::Quaterniond(Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX()));

    // WHEN: we relax them with relative orientation
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relax3dPointMeasurements(graph, np, edges);

    // THEN: it should put them back into the original orientation
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 1e-8)
            << i << ": " << np[i].orientation.coeffs().transpose() << std::endl
            << "g: " << ground_ori[i].coeffs().transpose();
}

TEST(relax, measurement_3_images_plane)
{
    // GIVEN: a graph, 3 images with edges between them all, then with their rotation disturbed
    MeasurementGraph graph;
    std::vector<NodePose> np;
    CameraModel model;
    model.focal_length_pixels = 600;
    model.principle_point << 400, 300;
    model.pixels_cols = 800;
    model.pixels_rows = 600;
    model.projection_type = opencalibration::ProjectionType::PLANAR;

    size_t id[3];
    Eigen::Quaterniond ori0(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond ori1(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitY()));
    Eigen::Quaterniond ori2(Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitX()));
    Eigen::Quaterniond ground_ori[3]{ori0, ori1, ori2};
    Eigen::Vector3d pos0(9, 9, 9);
    Eigen::Vector3d pos1(11, 9, 9);
    Eigen::Vector3d pos2(11, 11, 9);
    Eigen::Vector3d ground_pos[3]{pos0, pos1, pos2};

    for (int i = 0; i < 3; i++)
    {
        image img;
        img.orientation = ground_ori[i];
        img.position = ground_pos[i];
        img.model = model;
        id[i] = graph.addNode(std::move(img));
        np.emplace_back(NodePose{id[i], ground_ori[i], ground_pos[i]});
    }
    // 3d points that they all see

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vec3d;
    vec3d.reserve(100);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            vec3d.emplace_back(i + 5, j + 5, -10 + 1e-3 * i + 1e-2 * j);
        }
    }

    size_t edge_id[3];
    for (size_t i = 0; i < 3; i++)
    {
        camera_relations relation;
        size_t index[2] = {i, (i + 1) % 3};
        for (const Eigen::Vector3d &p : vec3d)
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

    // add some noise to the orientations
    np[0].orientation = Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitY());
    np[1].orientation = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ());
    np[2].orientation = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX());

    // WHEN: we relax them with relative orientation
    std::unordered_set<size_t> edges{edge_id[0], edge_id[1], edge_id[2]};
    relaxGroundPlaneMeasurements(graph, np, edges);

    // THEN: it should put them back into the original orientation
    for (int i = 0; i < 3; i++)
        EXPECT_LT(Eigen::AngleAxisd(np[i].orientation.inverse() * ground_ori[i]).angle(), 1e-8)
            << i << ": " << np[i].orientation.coeffs().transpose() << std::endl
            << "g: " << ground_ori[i].coeffs().transpose();
}
