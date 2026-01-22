#include <opencalibration/io/checkpoint.hpp>

#include <opencalibration/io/deserialize.hpp>
#include <opencalibration/io/serialize.hpp>

#include <spdlog/spdlog.h>

#define RAPIDJSON_HAS_STDSTRING 1
#define RAPIDJSON_WRITE_DEFAULT_FLAGS kWriteNanAndInfFlag
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <filesystem>
#include <fstream>

namespace opencalibration
{

namespace
{

bool saveMetadata(const CheckpointData &data, const std::filesystem::path &checkpoint_dir)
{
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);

    writer.StartObject();

    writer.Key("version");
    writer.Int(1);

    writer.Key("state");
    writer.String(pipelineStateToString(data.state).c_str());

    writer.Key("state_run_count");
    writer.Uint64(data.state_run_count);

    writer.Key("origin_latitude");
    writer.Double(data.origin_latitude);

    writer.Key("origin_longitude");
    writer.Double(data.origin_longitude);

    writer.Key("surface_count");
    writer.Uint64(data.surfaces.size());

    writer.EndObject();

    std::ofstream out(checkpoint_dir / "metadata.json");
    if (!out.is_open())
    {
        spdlog::error("Failed to open metadata.json for writing");
        return false;
    }
    out << buffer.GetString();
    return true;
}

bool loadMetadata(CheckpointData &data, const std::filesystem::path &checkpoint_dir, size_t &surface_count)
{
    std::ifstream in(checkpoint_dir / "metadata.json");
    if (!in.is_open())
    {
        spdlog::error("Failed to open metadata.json for reading");
        return false;
    }

    std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    rapidjson::Document doc;
    if (doc.Parse(json.c_str()).HasParseError())
    {
        spdlog::error("Failed to parse metadata.json");
        return false;
    }

    if (!doc.HasMember("version") || doc["version"].GetInt() != 1)
    {
        spdlog::error("Unsupported checkpoint version");
        return false;
    }

    if (doc.HasMember("state"))
    {
        data.state = stringToPipelineState(doc["state"].GetString());
    }

    if (doc.HasMember("state_run_count"))
    {
        data.state_run_count = doc["state_run_count"].GetUint64();
    }

    if (doc.HasMember("origin_latitude"))
    {
        data.origin_latitude = doc["origin_latitude"].GetDouble();
    }

    if (doc.HasMember("origin_longitude"))
    {
        data.origin_longitude = doc["origin_longitude"].GetDouble();
    }

    if (doc.HasMember("surface_count"))
    {
        surface_count = doc["surface_count"].GetUint64();
    }

    return true;
}

bool savePointCloud(const point_cloud &cloud, const std::filesystem::path &filepath)
{
    std::ofstream out(filepath);
    if (!out.is_open())
    {
        spdlog::error("Failed to open {} for writing", filepath.string());
        return false;
    }

    for (const auto &point : cloud)
    {
        out << point.x() << "," << point.y() << "," << point.z() << "\n";
    }
    return true;
}

bool loadPointCloud(point_cloud &cloud, const std::filesystem::path &filepath)
{
    std::ifstream in(filepath);
    if (!in.is_open())
    {
        spdlog::error("Failed to open {} for reading", filepath.string());
        return false;
    }

    cloud.clear();
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty())
            continue;

        size_t pos1 = line.find(',');
        size_t pos2 = line.find(',', pos1 + 1);
        if (pos1 == std::string::npos || pos2 == std::string::npos)
        {
            continue;
        }

        double x = std::stod(line.substr(0, pos1));
        double y = std::stod(line.substr(pos1 + 1, pos2 - pos1 - 1));
        double z = std::stod(line.substr(pos2 + 1));
        cloud.push_back(Eigen::Vector3d(x, y, z));
    }
    return true;
}

} // namespace

bool saveCheckpoint(const CheckpointData &data, const std::string &checkpoint_dir)
{
    std::filesystem::path dir(checkpoint_dir);

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
    {
        spdlog::error("Failed to create checkpoint directory: {}", ec.message());
        return false;
    }

    if (!saveMetadata(data, dir))
    {
        return false;
    }

    std::ofstream graph_out(dir / "graph.json");
    if (!graph_out.is_open())
    {
        spdlog::error("Failed to open graph.json for writing");
        return false;
    }
    if (!serialize(data.graph, graph_out))
    {
        spdlog::error("Failed to serialize graph");
        return false;
    }
    graph_out.close();

    for (size_t i = 0; i < data.surfaces.size(); i++)
    {
        const auto &surface = data.surfaces[i];

        if (surface.mesh.size_nodes() > 0)
        {
            std::string mesh_filename = "surface_" + std::to_string(i) + ".ply";
            std::ofstream mesh_out(dir / mesh_filename);
            if (!mesh_out.is_open())
            {
                spdlog::error("Failed to open {} for writing", mesh_filename);
                return false;
            }
            if (!serialize(surface.mesh, mesh_out))
            {
                spdlog::error("Failed to serialize mesh {}", i);
                return false;
            }
            mesh_out.close();
        }

        for (size_t j = 0; j < surface.cloud.size(); j++)
        {
            std::string cloud_filename = "pointcloud_" + std::to_string(i) + "_" + std::to_string(j) + ".xyz";
            if (!savePointCloud(surface.cloud[j], dir / cloud_filename))
            {
                return false;
            }
        }

        std::string cloud_count_filename = "surface_" + std::to_string(i) + "_cloudcount.txt";
        std::ofstream count_out(dir / cloud_count_filename);
        if (count_out.is_open())
        {
            count_out << surface.cloud.size();
        }
    }

    spdlog::info("Checkpoint saved to {}", checkpoint_dir);
    return true;
}

bool loadCheckpoint(const std::string &checkpoint_dir, CheckpointData &data)
{
    std::filesystem::path dir(checkpoint_dir);

    if (!std::filesystem::exists(dir))
    {
        spdlog::error("Checkpoint directory does not exist: {}", checkpoint_dir);
        return false;
    }

    size_t surface_count = 0;
    if (!loadMetadata(data, dir, surface_count))
    {
        return false;
    }

    std::ifstream graph_in(dir / "graph.json");
    if (!graph_in.is_open())
    {
        spdlog::error("Failed to open graph.json for reading");
        return false;
    }
    std::string graph_json((std::istreambuf_iterator<char>(graph_in)), std::istreambuf_iterator<char>());
    graph_in.close();

    if (!deserialize(graph_json, data.graph))
    {
        spdlog::error("Failed to deserialize graph");
        return false;
    }

    data.surfaces.clear();
    data.surfaces.resize(surface_count);

    for (size_t i = 0; i < surface_count; i++)
    {
        auto &surface = data.surfaces[i];

        std::string mesh_filename = "surface_" + std::to_string(i) + ".ply";
        std::filesystem::path mesh_path = dir / mesh_filename;
        if (std::filesystem::exists(mesh_path))
        {
            std::ifstream mesh_in(mesh_path);
            if (mesh_in.is_open())
            {
                if (!deserialize(mesh_in, surface.mesh))
                {
                    spdlog::warn("Failed to deserialize mesh {}", i);
                }
            }
        }

        std::string cloud_count_filename = "surface_" + std::to_string(i) + "_cloudcount.txt";
        std::filesystem::path count_path = dir / cloud_count_filename;
        size_t cloud_count = 0;
        if (std::filesystem::exists(count_path))
        {
            std::ifstream count_in(count_path);
            if (count_in.is_open())
            {
                count_in >> cloud_count;
            }
        }

        surface.cloud.resize(cloud_count);
        for (size_t j = 0; j < cloud_count; j++)
        {
            std::string cloud_filename = "pointcloud_" + std::to_string(i) + "_" + std::to_string(j) + ".xyz";
            std::filesystem::path cloud_path = dir / cloud_filename;
            if (std::filesystem::exists(cloud_path))
            {
                if (!loadPointCloud(surface.cloud[j], cloud_path))
                {
                    spdlog::warn("Failed to load point cloud {} for surface {}", j, i);
                }
            }
        }
    }

    spdlog::info("Checkpoint loaded from {}", checkpoint_dir);
    return true;
}

bool validateCheckpoint(const std::string &checkpoint_dir)
{
    std::filesystem::path dir(checkpoint_dir);

    if (!std::filesystem::exists(dir))
    {
        return false;
    }

    if (!std::filesystem::exists(dir / "metadata.json"))
    {
        return false;
    }

    if (!std::filesystem::exists(dir / "graph.json"))
    {
        return false;
    }

    return true;
}

} // namespace opencalibration
