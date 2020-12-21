#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/io/serialize.hpp>

#include <spdlog/spdlog.h>

#include <omp.h>

#include <filesystem>
#include <iostream>
#include <thread>

using namespace opencalibration;
using namespace std::chrono_literals;

int main(int argc, char *argv[])
{
    spdlog::set_level(spdlog::level::err);
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
    if (argc > 1)
    {
        std::string path = argv[1];
        for (const auto &entry : std::filesystem::directory_iterator(path))
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

    std::cerr << out << std::endl;
}
