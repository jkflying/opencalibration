#include <opencalibration/io/serialize.hpp>
#include <opencalibration/io/deserialize.hpp>

#include <opencalibration/pipeline/pipeline.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;
using namespace std::chrono_literals;


TEST(serialize_graph, empty_graph)
{
    std::string serialized = serialize(MeasurementGraph());
    EXPECT_EQ(serialized, "{\n"
                          "    \"version\": 1,\n"
                          "    \"nodes\": {},\n"
                          "    \"edges\": {}\n"
                          "}");
}


TEST(serialize_graph, one_image)
{
    Pipeline p(1);
    p.add(TEST_DATA_DIR "P2530253.JPG");

    while (p.getStatus() != Pipeline::Status::COMPLETE)
    {
        std::this_thread::sleep_for(1ms);
    }

    std::string serialized = serialize(p.getGraph());
    MeasurementGraph g;
    deserialize(serialized, g);
    EXPECT_EQ(g, p.getGraph());
}
