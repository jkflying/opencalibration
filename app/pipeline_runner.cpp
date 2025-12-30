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
    p.set_generate_thumbnails(generate_thumbnails);
    p.set_thumbnail_filenames(thumbnail_file, source_file, overlap_file);
    p.set_geotiff_filename(geotiff_file);
    p.set_dsm_filename(dsm_file);

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
        case PipelineState::GENERATE_THUMBNAIL:
        case PipelineState::GENERATE_GEOTIFF:
        case PipelineState::COMPLETE: {
            std::cout << std::endl;
            break;
        }
        }
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

    while (p.getState() != PipelineState::COMPLETE)
    {
        p.iterateOnce();
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
