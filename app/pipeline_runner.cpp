#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/io/serialize.hpp>

#include <spdlog/sinks/basic_file_sink.h>
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
    std::string output_file = "";
    std::string log_file = "";
    int batch_size = omp_get_max_threads();
    bool printHelp = false;

    CommandLine args("Run the opencalibration pipeline from the command line");
    args.addArgument({"-i", "--input"}, &input_dir, "Input directory of images");
    args.addArgument({"-d", "--debug"}, &debug_level, "none=0, critical=1, error=2, warn=3, info=4, debug=5");
    args.addArgument({"-l", "--log-file"}, &log_file, "Output logging file, overwrites existing files");
    args.addArgument({"-o", "--output"}, &output_file, "Output geojson file");
    args.addArgument({"-b", "--batch-size"}, &batch_size, "Processing batch size");
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
    std::string log_level_str = "err";
    switch (debug_level)
    {
    case 0:
        level = spdlog::level::off;
        log_level_str = "off";
        break;
    case 1:
        level = spdlog::level::critical;
        log_level_str = "critical";
        break;
    case 2:
        level = spdlog::level::err;
        log_level_str = "err";
        break;
    case 3:
        level = spdlog::level::warn;
        log_level_str = "warn";
        break;
    case 4:
        level = spdlog::level::info;
        log_level_str = "info";
        break;
    case 5:
        level = spdlog::level::debug;
        log_level_str = "debug";
        break;
    }
    spdlog::set_level(level);
    if (log_file.size() > 0)
    {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file);
        spdlog::default_logger()->sinks().push_back(std::move(file_sink));
    }

    spdlog::info("Log level set to {}", log_level_str);
    spdlog::info("Pipeline batch size set to {}", batch_size);

    Pipeline p(batch_size);

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

    if (output_file.size() > 0)
    {
        auto to_wgs84 = [&p](const Eigen::Vector3d &local) { return p.getCoord().toWGS84(local); };
        std::string out = toVisualizedGeoJson(p.getGraph(), to_wgs84);

        std::ofstream output;
        output.open(output_file, std::ios::out | std::ios::trunc);
        output << out;
        output.close();
    }
    std::cout << "Complete!" << std::endl;
}
