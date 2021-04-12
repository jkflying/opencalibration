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
    relation.relative_rotation = Eigen::Quaterniond::Identity();
    //     std::cout << relation.relative_rotation.coeffs().transpose() << std::endl;
    relation.relative_translation << 1, 0, 0;
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
