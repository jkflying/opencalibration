#include <opencalibration/io/deserialize.hpp>
#include <opencalibration/io/serialize.hpp>

#include <opencalibration/pipeline/pipeline.hpp>

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
