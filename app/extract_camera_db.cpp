#include <opencalibration/extract/camera_database.hpp>
#include <opencalibration/io/checkpoint.hpp>

#include <spdlog/spdlog.h>

#include "CommandLine.hpp"

#include <iostream>

using namespace opencalibration;

int main(int argc, char *argv[])
{
    std::string checkpoint_load;
    std::string database_path = CameraDatabase::defaultPath();
    std::string notes;
    bool printHelp = false;

    CommandLine args("Extract camera model parameters from a checkpoint into camera_database.json");
    args.addArgument({"-cl", "--checkpoint-load"}, &checkpoint_load, "Checkpoint directory to load (required)");
    args.addArgument({"-db", "--database"}, &database_path, "Path to camera_database.json");
    args.addArgument({"-n", "--notes"}, &notes, "Notes to attach to extracted entries");
    args.addArgument({"-h", "--help"}, &printHelp, "Show help");

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

    if (checkpoint_load.empty())
    {
        std::cerr << "Error: --checkpoint-load is required" << std::endl;
        args.printHelp();
        return -1;
    }

    spdlog::set_level(spdlog::level::info);

    CheckpointData data;
    if (!loadCheckpoint(checkpoint_load, data))
    {
        spdlog::error("Failed to load checkpoint from {}", checkpoint_load);
        return -1;
    }
    spdlog::info("Loaded checkpoint with {} nodes", data.graph.size_nodes());

    if (!updateDatabaseFromGraph(data.graph, database_path, notes))
    {
        return -1;
    }

    return 0;
}
