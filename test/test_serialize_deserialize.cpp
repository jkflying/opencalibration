#include <opencalibration/io/deserialize.hpp>
#include <opencalibration/io/serialize.hpp>
#include <opencalibration/surface/expand_mesh.hpp>

#include <opencalibration/pipeline/pipeline.hpp>

#include <fstream>
#include <gtest/gtest.h>

using namespace opencalibration;
using namespace std::chrono_literals;

TEST(serialize_graph, empty_graph)
{
    std::ostringstream serialized;
    EXPECT_TRUE(serialize(MeasurementGraph(), serialized));
    EXPECT_EQ(serialized.str(), "{\n"
                                "    \"version\": 1,\n"
                                "    \"nodes\": {},\n"
                                "    \"edges\": {}\n"
                                "}");
}

TEST(serialize_graph, one_image)
{
    Pipeline p(1);
    p.add({TEST_DATA_DIR "P2530253.JPG"});

    while (p.getState() != PipelineState::COMPLETE)
    {
        p.iterateOnce();
    }

    std::ostringstream serialized;
    EXPECT_TRUE(serialize(p.getGraph(), serialized));
    MeasurementGraph g;
    deserialize(serialized.str(), g);
    EXPECT_TRUE(g == p.getGraph());

    std::ostringstream dedeserialized;
    EXPECT_TRUE(serialize(g, dedeserialized));
    EXPECT_EQ(serialized.str(), dedeserialized.str());
}

TEST(serialize_graph, three_images)
{
    Pipeline p(1);
    p.add({TEST_DATA_DIR "P2530253.JPG", TEST_DATA_DIR "P2540254.JPG", TEST_DATA_DIR "P2550255.JPG"});

    while (p.getState() != PipelineState::COMPLETE)
    {
        p.iterateOnce();
    }

    std::ostringstream serialized;
    EXPECT_TRUE(serialize(p.getGraph(), serialized));
    MeasurementGraph g;
    deserialize(serialized.str(), g);
    EXPECT_TRUE(g == p.getGraph());

    std::ostringstream dedeserialized;
    EXPECT_TRUE(serialize(g, dedeserialized));
    EXPECT_EQ(serialized.str(), dedeserialized.str());
}

TEST(serialize_meshgraph, simple_surface)
{
    // GIVEN: a mesh
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0)};
    g = rebuildMesh(p, {surface_model{{}, g}});

    // WHEN: we write it to a string
    std::ostringstream osstream;
    EXPECT_TRUE(serialize(g, osstream));
    std::istringstream isstream(osstream.str());

    // THEN: it should be exaclty the same when we read it from the string
    MeshGraph deserialized;
    EXPECT_TRUE(deserialize(isstream, deserialized));
    EXPECT_EQ(g, deserialized);

    // AND: when we write it to a string again it should be the same again
    std::ostringstream osstream2;
    EXPECT_TRUE(serialize(deserialized, osstream2));
    EXPECT_EQ(osstream.str(), osstream2.str());

    // AND: write a copy to file to help with debugging
    {
        std::ofstream ofstream;
        ofstream.open(TEST_DATA_OUTPUT_DIR "simple.ply");
        EXPECT_TRUE(serialize(g, ofstream));
        ofstream.close();
    }
}

TEST(serialize_meshgraph, complex_uniform_surface)
{
    MeshGraph g;
    point_cloud p{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 0, 0.5), Eigen::Vector3d(1, 1, 0.3),
                  Eigen::Vector3d(0, 1, -0.5)};

    g = rebuildMesh(p, {surface_model{{}, g}});

    // WHEN: we write it to a string
    std::ostringstream osstream;
    EXPECT_TRUE(serialize(g, osstream));
    std::istringstream isstream(osstream.str());

    // THEN: it should be exaclty the same when we read it from the string
    MeshGraph deserialized;
    EXPECT_TRUE(deserialize(isstream, deserialized));
    EXPECT_EQ(g, deserialized);

    // AND: when we write it to a string again it should be the same again
    std::ostringstream osstream2;
    EXPECT_TRUE(serialize(deserialized, osstream2));
    EXPECT_EQ(osstream.str(), osstream2.str());

    // AND: write a copy to file to help with debugging
    {
        std::ofstream ofstream;
        ofstream.open(TEST_DATA_OUTPUT_DIR "surface.ply");
        EXPECT_TRUE(serialize(g, ofstream));
        ofstream.close();
    }
}
