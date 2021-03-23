#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/io/serialize.hpp>

#include <spdlog/spdlog.h>

#include "CommandLine.hpp"

#include <omp.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

using namespace opencalibration;
using namespace std::chrono_literals;

int main(int argc, char *argv[])
{

    std::string input_dir = "";
    uint32_t debug_level = 2;
    std::string output_file = "output.geojson";
    bool printHelp = false;

    CommandLine args("Run the opencalibration pipeline from the command line");
    args.addArgument({"-i", "--input"}, &input_dir, "Input directory of images");
    args.addArgument({"-d", "--debug"}, &debug_level, "none=0, critical=1, error=2, warn=3, info=4, debug=5");
    args.addArgument({"-o", "--output"}, &output_file, "Output geojson file");
    args.addArgument({"-h", "--help"}, &printHelp, "You must specify at least an input file");

    try
    {
        args.parse(argc, argv);
    }
    catch (std::runtime_error const &e)
    {
        std::cout << e.what() << std::endl;
        return -1;
    }

    if (printHelp)
    {
        args.printHelp();
        return 0;
    }

    auto level = spdlog::level::err;
    switch (debug_level)
    {
    case 0:
        level = spdlog::level::off;
        break;
    case 1:
        level = spdlog::level::critical;
        break;
    case 2:
        level = spdlog::level::err;
        break;
    case 3:
        level = spdlog::level::warn;
        break;
    case 4:
        level = spdlog::level::info;
        break;
    case 5:
        level = spdlog::level::debug;
        break;
    }
    spdlog::set_level(level);

    int pipeline_width = omp_get_max_threads();

    spdlog::info("Pipeline width set to {}", pipeline_width);

    Pipeline p(pipeline_width);

    p.set_callback([](const Pipeline::StepCompletionInfo &info) {
        std::cout << "Progress: " << info.images_loaded << " / " << (info.images_loaded + info.queue_size_remaining)
                  << " "
                  << " loaded: " << info.loaded_ids.get().size() << " linked: " << info.linked_ids.get().size()
                  << " relaxed: " << info.relaxed_ids.get().size() << std::endl;
    });

    std::vector<std::string> files;
    if (input_dir.size() > 0)
    {
        for (const auto &entry : std::filesystem::directory_iterator(input_dir))
        {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    p.add(files);

    while (p.getStatus() != Pipeline::Status::COMPLETE)
    {
        std::this_thread::sleep_for(1ms);
    }

    auto to_wgs84 = [&p](const Eigen::Vector3d &local) { return p.getCoord().toWGS84(local); };
    std::string out = toVisualizedGeoJson(p.getGraph(), to_wgs84);

    std::fstream output;
    output.open(output_file, std::ios::out | std::ios::trunc);
    output << out;
    output.close();
    std::cout << "Complete!" << std::endl;
}
