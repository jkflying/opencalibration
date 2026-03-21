#include <opencalibration/extract/camera_database.hpp>
#include <opencalibration/io/checkpoint.hpp>

#include <spdlog/spdlog.h>

#include "CommandLine.hpp"

#include <iostream>

using namespace opencalibration;

int main(int argc, char *argv[])
{
    std::string checkpoint_restore;
    std::string database_path = CameraDatabase::defaultPath();
    std::string notes;
    bool print_help = false;

    CommandLine args("Extract camera model parameters from a checkpoint into camera_database.json");
    args.addArgument({"-r", "--checkpoint-restore"}, &checkpoint_restore, "Checkpoint directory to restore (required)");
    args.addArgument({"--database"}, &database_path, "Path to camera_database.json");
    args.addArgument({"-n", "--notes"}, &notes, "Notes to attach to extracted entries");
    args.addArgument({"-h", "--help"}, &print_help, "Show help");

    try
    {
        args.parse(argc, argv);
    }
    catch (std::runtime_error const &e)
    {
        std::cout << e.what() << std::endl;
        return -1;
    }

    if (print_help)
    {
        args.printHelp();
        return 0;
    }

    if (checkpoint_restore.empty())
    {
        std::cerr << "Error: --checkpoint-restore is required" << std::endl;
        args.printHelp();
        return -1;
    }

    spdlog::set_level(spdlog::level::info);

    CheckpointData data;
    if (!loadCheckpoint(checkpoint_restore, data))
    {
        spdlog::error("Failed to load checkpoint from {}", checkpoint_restore);
        return -1;
    }
    spdlog::info("Loaded checkpoint with {} nodes", data.graph.size_nodes());

    if (!updateDatabaseFromGraph(data.graph, database_path, notes))
    {
        return -1;
    }

    return 0;
}
