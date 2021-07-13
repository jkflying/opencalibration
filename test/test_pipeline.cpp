#include <opencalibration/pipeline/pipeline.hpp>

#include <gtest/gtest.h>

#include <chrono>

using namespace opencalibration;
using namespace std::chrono_literals;

TEST(pipeline, constructs_with_lots_of_threads)
{
    Pipeline p(100);
    std::this_thread::sleep_for(1ms);
}

TEST(pipeline, processes_4_images)
{

    // GIVEN: a pipeline and paths
    Pipeline p(2);
    std::string path1 = TEST_DATA_DIR "P2530253.JPG";
    std::string path2 = TEST_DATA_DIR "P2540254.JPG";
    std::string path3 = TEST_DATA_DIR "P2550255.JPG";
    std::string path4 = TEST_DATA_DIR "P2560256.JPG";

    // WHEN: we add the paths
    p.add({path1, path2, path3, path4});

    // THEN: after some time they should all be processed
    while (p.getState() != PipelineState::COMPLETE)
    {
        p.iterateOnce();
    }
}
