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
    relaxDecompositions(graph, np);

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
    relaxDecompositions(graph, np);

    // THEN: it should be pointing downwards, with other axes close to the original
    EXPECT_LT((np[0].orientation.coeffs() - Eigen::Vector4d(1, 0, 0, 0)).norm(), 1e-3)
        << np[0].orientation.coeffs().transpose();
}

TEST(relax, prior_2_images)
{
    // GIVEN: a graph, with 2 images with relative orientation as identity
    MeasurementGraph graph;
    std::vector<NodePose> np;

    Eigen::Quaterniond ori(Eigen::AngleAxisd(M_PI_4, Eigen::Vector3d::UnitX()));
    Eigen::Vector3d pos(9, 9, 9);
    image img;
    img.orientation = ori;
    img.position = pos;

    const size_t id = graph.addNode(std::move(img));
    np.emplace_back(NodePose{id, ori, pos});

    ori = Eigen::Quaterniond(Eigen::AngleAxisd(-M_PI_4, Eigen::Vector3d::UnitY()));
    pos << 0, 9, 9;

    image img2;
    img2.orientation = ori;
    img2.position = pos;
    const size_t id2 = graph.addNode(std::move(img2));
    np.emplace_back(NodePose{id2, ori, pos});

    camera_relations relation;
    relation.relative_rotation = Eigen::Quaterniond::Identity();
    //     std::cout << relation.relative_rotation.coeffs().transpose() << std::endl;
    relation.relative_translation << -1, 0, 0;
    graph.addEdge(std::move(relation), id, id2);

    // WHEN: we relax the relative orientations
    relaxDecompositions(graph, np);

    // THEN: it should be pointing downwards, with other axes close to the original
    EXPECT_LT(Eigen::AngleAxisd(np[0].orientation.inverse() *
                                Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX())))
                  .angle(),
              1e-3)
        << np[0].orientation.coeffs().transpose();
    EXPECT_LT(Eigen::AngleAxisd(np[1].orientation.inverse() *
                                Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX())))
                  .angle(),
              1e-3)
        << np[1].orientation.coeffs().transpose();
}
