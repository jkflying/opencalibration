#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/io/serialize.hpp>

#include <filesystem>
#include <thread>

#include <iostream>

using namespace opencalibration;
using namespace std::chrono_literals;

int main(int argc, char *argv[])
{
    Pipeline p(1);
    std::vector<std::string> files;

    if (argc > 1)
    {
        std::string path = argv[1];
        for (const auto &entry : std::filesystem::directory_iterator(path))
        {
            files.push_back(entry.path());
        }
    }

    std::random_shuffle(files.begin(), files.end());
    p.add(files);

    while (p.getStatus() != Pipeline::Status::COMPLETE)
    {
        std::this_thread::sleep_for(1ms);
    }

    std::string out = serialize(p.getGraph());

    std::cout << out << std::endl;
}
