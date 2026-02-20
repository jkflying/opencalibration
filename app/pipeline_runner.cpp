#include <opencalibration/extract/camera_database.hpp>
#include <opencalibration/performance/performance.hpp>
#include <opencalibration/pipeline/pipeline.hpp>

#include <opencalibration/io/saveXYZ.hpp>
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

    EnablePerformanceCounters(true);
    std::string input_dir = "";
    uint32_t debug_level = 2;
    std::string output_file = "";
    std::string serialized_graph_file = "";
    std::string pointcloud_file = "";
    std::string mesh_file = "";
    std::string log_file = "";
    int batch_size = omp_get_max_threads();
    bool generate_thumbnails = true;
    std::string thumbnail_file = "";
    std::string source_file = "";
    std::string overlap_file = "";
    std::string geotiff_file = "";
    std::string dsm_file = "";
    std::string textured_mesh_file = "";
    double ortho_max_megapixels = 0.0;
    std::string checkpoint_save = "";
    std::string checkpoint_load = "";
    std::string resume_from = "";
    bool update_camera_db = false;
    bool skip_mesh_refinement = false;
    bool skip_initial_global_relax = false;
    bool skip_camera_intrinsics = false;
    bool skip_final_global_relax = false;
    bool printHelp = false;

    CommandLine args("Run the opencalibration pipeline from the command line");
    args.addArgument({"-i", "--input"}, &input_dir, "Input directory of images");
    args.addArgument({"-d", "--debug"}, &debug_level, "none=0, critical=1, error=2, warn=3, info=4, debug=5");
    args.addArgument({"-l", "--log-file"}, &log_file, "Output logging file, overwrites existing files");
    args.addArgument({"-g", "--geojson"}, &output_file, "Output geojson file");
    args.addArgument({"-p", "--pointcloud-file"}, &pointcloud_file, "Output pointcloud file");
    args.addArgument({"-m", "--mesh-file"}, &mesh_file, "Output mesh PLY file");
    args.addArgument({"-s", "--serialize"}, &serialized_graph_file, "Output serialized camera graph file");
    args.addArgument({"-b", "--batch-size"}, &batch_size, "Processing batch size");
    args.addArgument({"-t", "--thumbnails"}, &generate_thumbnails, "Generate thumbnails (default: true)");
    args.addArgument({"--thumbnail-file"}, &thumbnail_file, "Output thumbnail image file");
    args.addArgument({"--source-file"}, &source_file, "Output source index image file");
    args.addArgument({"--overlap-file"}, &overlap_file, "Output overlap count image file");
    args.addArgument({"--ortho-geotiff"}, &geotiff_file, "Output full-resolution georeferenced GeoTIFF orthomosaic");
    args.addArgument({"--dsm-geotiff"}, &dsm_file, "Output Digital Surface Model (DSM) GeoTIFF");
    args.addArgument({"--textured-mesh"}, &textured_mesh_file, "Output textured OBJ mesh (generates .obj, .mtl, .jpg)");
    args.addArgument({"--ortho-max-megapixels"}, &ortho_max_megapixels,
                     "Maximum orthomosaic output size in megapixels (0 = unlimited)");
    args.addArgument({"-cs", "--checkpoint-save"}, &checkpoint_save, "Save checkpoint to directory after processing");
    args.addArgument({"-cl", "--checkpoint-load"}, &checkpoint_load, "Load and resume from checkpoint directory");
    args.addArgument({"--update-camera-db"}, &update_camera_db,
                     "Update data/camera_database.json with optimized parameters after pipeline completes");
    args.addArgument({"--resume-from"}, &resume_from,
                     "Resume from specific stage (INITIAL_GLOBAL_RELAX, CAMERA_PARAMETER_RELAX, FINAL_GLOBAL_RELAX, "
                     "GENERATE_THUMBNAIL, GENERATE_LAYERS, COLOR_BALANCE, BLEND_LAYERS)");
    args.addArgument({"--skip-mesh-refinement"}, &skip_mesh_refinement,
                     "Skip the mesh refinement stage (uses grid mesh instead of adaptive refinement)");
    args.addArgument({"--skip-initial-global-relax"}, &skip_initial_global_relax,
                     "Skip the initial global relaxation stage");
    args.addArgument({"--skip-camera-intrinsics"}, &skip_camera_intrinsics,
                     "Skip camera intrinsics optimization (focal length, distortion, principal point)");
    args.addArgument({"--skip-final-global-relax"}, &skip_final_global_relax, "Skip the final global relaxation stage");
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

    if (ortho_max_megapixels < 0.0)
    {
        std::cerr << "--ortho-max-megapixels must be >= 0" << std::endl;
        return -1;
    }

    if (!textured_mesh_file.empty() && geotiff_file.empty())
    {
        std::cerr << "--textured-mesh requires --ortho-geotiff to be set" << std::endl;
        return -1;
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
    if (ortho_max_megapixels > 0.0)
    {
        spdlog::info("Orthomosaic max output set to {} MP", ortho_max_megapixels);
    }

    if (!resume_from.empty() && !Pipeline::fromString(resume_from))
    {
        spdlog::error("Unrecognized --resume-from state: '{}'. Valid states: INITIAL_GLOBAL_RELAX, "
                      "CAMERA_PARAMETER_RELAX, FINAL_GLOBAL_RELAX, GENERATE_THUMBNAIL, GENERATE_LAYERS, "
                      "COLOR_BALANCE, BLEND_LAYERS",
                      resume_from);
        return -1;
    }

    Pipeline p(batch_size);
    p.set_generate_thumbnails(generate_thumbnails);
    p.set_thumbnail_filenames(thumbnail_file, source_file, overlap_file);
    p.set_geotiff_filename(geotiff_file);
    p.set_dsm_filename(dsm_file);
    p.set_textured_mesh_filename(textured_mesh_file);
    p.set_orthomosaic_max_megapixels(ortho_max_megapixels);
    p.set_skip_mesh_refinement(skip_mesh_refinement);
    p.set_skip_initial_global_relax(skip_initial_global_relax);
    p.set_skip_camera_param_relax(skip_camera_intrinsics);
    p.set_skip_final_global_relax(skip_final_global_relax);

    if (!checkpoint_load.empty())
    {
        spdlog::info("Loading checkpoint from {}", checkpoint_load);
        if (!p.loadCheckpoint(checkpoint_load))
        {
            spdlog::error("Failed to load checkpoint from {}", checkpoint_load);
            return -1;
        }
        spdlog::info("Loaded checkpoint, current state: {}", Pipeline::toString(p.getState()));

        if (!resume_from.empty())
        {
            auto target_state_opt = Pipeline::fromString(resume_from);
            if (!target_state_opt)
            {
                spdlog::error("Unknown state: {}", resume_from);
                return -1;
            }
            if (!p.resumeFromState(*target_state_opt))
            {
                spdlog::error("Failed to resume from state {}", resume_from);
                return -1;
            }
        }
    }

    p.set_callback([](const Pipeline::StepCompletionInfo &info) {
        std::cout << Pipeline::toString(info.state);
        size_t total_relaxed = 0;
        for (const auto &ids_group : info.relaxed_ids.get())
            total_relaxed += ids_group.size();

        switch (info.state)
        {
        case PipelineState::INITIAL_PROCESSING: {
            std::cout << ": " << info.images_loaded << " / " << (info.images_loaded + info.queue_size_remaining) << " "
                      << " loaded: " << info.loaded_ids.get().size() << " linked: " << info.linked_ids.get().size()
                      << " relaxed: " << total_relaxed << std::endl;
            break;
        }
        case PipelineState::INITIAL_GLOBAL_RELAX:
        case PipelineState::CAMERA_PARAMETER_RELAX:
        case PipelineState::FINAL_GLOBAL_RELAX: {
            std::cout << ": iteration: " << info.state_iteration << " relaxed: " << total_relaxed << std::endl;
            break;
        }
        case PipelineState::MESH_REFINEMENT: {
            std::cout << ": refining mesh based on point density" << std::endl;
            break;
        }
        case PipelineState::GENERATE_THUMBNAIL:
        case PipelineState::GENERATE_LAYERS:
        case PipelineState::COLOR_BALANCE:
        case PipelineState::BLEND_LAYERS:
        case PipelineState::COMPLETE: {
            std::cout << std::endl;
            break;
        }
        }
    });

    if (checkpoint_load.empty())
    {
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
    }
    else if (input_dir.size() > 0)
    {
        spdlog::warn("Input directory ignored when loading from checkpoint");
    }

    PipelineState previous_state = p.getState();
    while (p.getState() != PipelineState::COMPLETE)
    {
        p.iterateOnce();

        if (!checkpoint_save.empty() && p.getState() != previous_state)
        {
            spdlog::info("Saving checkpoint to {} (state: {})", checkpoint_save, Pipeline::toString(p.getState()));
            if (!p.saveCheckpoint(checkpoint_save))
            {
                spdlog::error("Failed to save checkpoint to {}", checkpoint_save);
            }
            previous_state = p.getState();
        }
    }

    if (!checkpoint_save.empty())
    {
        spdlog::info("Saving final checkpoint to {}", checkpoint_save);
        if (!p.saveCheckpoint(checkpoint_save))
        {
            spdlog::error("Failed to save checkpoint to {}", checkpoint_save);
        }
        else
        {
            std::cout << "Checkpoint saved to " << checkpoint_save << std::endl;
        }
    }

    if (update_camera_db)
    {
        const auto &camera_db_path = CameraDatabase::defaultPath();
        spdlog::info("Updating camera database at {}", camera_db_path);
        if (!updateDatabaseFromGraph(p.getGraph(), camera_db_path))
        {
            spdlog::error("Failed to update camera database");
        }
    }

    if (output_file.size() > 0)
    {
        auto to_wgs84 = [&p](const Eigen::Vector3d &local) { return p.getCoord().toWGS84(local); };

        std::ofstream output;
        output.open(output_file, std::ios::binary);
        toVisualizedGeoJson(p.getGraph(), to_wgs84, output);
        output.close();
    }

    if (serialized_graph_file.size() > 0)
    {
        std::ofstream output;
        output.open(serialized_graph_file, std::ios::binary);
        serialize(p.getGraph(), output);
        output.close();
    }

    if (pointcloud_file.size() > 0 || mesh_file.size() > 0)
    {
        const auto &surface = p.getSurfaces();
        if (pointcloud_file.size() > 0)
        {
            std::ofstream output;
            output.open(pointcloud_file, std::ios::binary);
            toXYZ(surface, output, filterOutliers(surface));
            output.close();
        }
        if (mesh_file.size() > 0)
        {
            for (size_t i = 0; i < surface.size(); i++)
            {
                if (surface[i].mesh.size_edges() == 0)
                    continue;

                std::ofstream output;
                std::string filename = mesh_file;
                if (surface.size() > 1)
                    filename += "_" + std::to_string(i) + ".ply";
                output.open(filename, std::ios::binary);
                serialize(surface[i].mesh, output);
                output.close();
            }
        }
    }

    std::cout << "Complete!" << std::endl;
    std::cout << TotalPerformanceSummary() << std::endl;
}
