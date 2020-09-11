#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/io/serialize.hpp>

#include <spdlog/spdlog.h>

#include <filesystem>
#include <thread>
#include <iostream>

using namespace opencalibration;
using namespace std::chrono_literals;

int main(int argc, char *argv[])
{
    spdlog::set_level(spdlog::level::critical);
    Pipeline p(4);
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

    auto to_wgs84 = [&p](const Eigen::Vector3d& local) { return p.getCoord().toWGS84(local); };
    std::string out = toVisualizedGeoJson(p.getGraph(), to_wgs84);

    std::cout << out << std::endl;
}
