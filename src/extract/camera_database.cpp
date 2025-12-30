#include <opencalibration/extract/camera_database.hpp>

#include <spdlog/spdlog.h>

#define RAPIDJSON_HAS_STDSTRING 1
#include <rapidjson/document.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace
{
std::string toLower(const std::string &s)
{
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) { return std::tolower(c); });
    return result;
}
} // namespace

namespace opencalibration
{

CameraDatabase &CameraDatabase::instance()
{
    static CameraDatabase db;
    return db;
}

bool CameraDatabase::load(const std::string &path)
{
    std::lock_guard<std::mutex> lock(_mutex);

    if (_loaded)
    {
        return true;
    }

    std::ifstream file(path);
    if (!file.is_open())
    {
        spdlog::warn("Camera database not found at: {}", path);
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    rapidjson::Document doc;
    doc.Parse(json.c_str());

    if (doc.HasParseError())
    {
        spdlog::error("Failed to parse camera database JSON: error at offset {}", doc.GetErrorOffset());
        return false;
    }

    if (!doc.IsObject() || !doc.HasMember("version") || !doc.HasMember("cameras"))
    {
        spdlog::error("Camera database has invalid structure");
        return false;
    }

    int version = doc["version"].GetInt();
    if (version != 1)
    {
        spdlog::error("Unsupported camera database version: {}", version);
        return false;
    }

    const auto &cameras = doc["cameras"].GetArray();
    _entries.reserve(cameras.Size());

    for (const auto &cam : cameras)
    {
        CameraDBEntry entry;

        if (cam.HasMember("make"))
            entry.make = cam["make"].GetString();
        if (cam.HasMember("model"))
            entry.model = cam["model"].GetString();
        if (cam.HasMember("lens_model"))
            entry.lens_model = cam["lens_model"].GetString();

        if (cam.HasMember("sensor_width_px"))
            entry.sensor_width_px = cam["sensor_width_px"].GetUint64();
        if (cam.HasMember("sensor_height_px"))
            entry.sensor_height_px = cam["sensor_height_px"].GetUint64();

        if (cam.HasMember("radial_distortion"))
        {
            const auto &rd = cam["radial_distortion"].GetArray();
            for (size_t i = 0; i < std::min(rd.Size(), rapidjson::SizeType(3)); ++i)
            {
                entry.radial_distortion[i] = rd[i].GetDouble();
            }
        }

        if (cam.HasMember("tangential_distortion"))
        {
            const auto &td = cam["tangential_distortion"].GetArray();
            for (size_t i = 0; i < std::min(td.Size(), rapidjson::SizeType(2)); ++i)
            {
                entry.tangential_distortion[i] = td[i].GetDouble();
            }
        }

        if (cam.HasMember("principal_point_offset"))
        {
            const auto &pp = cam["principal_point_offset"].GetArray();
            for (size_t i = 0; i < std::min(pp.Size(), rapidjson::SizeType(2)); ++i)
            {
                entry.principal_point_offset[i] = pp[i].GetDouble();
            }
        }

        _entries.push_back(std::move(entry));
    }

    _loaded = true;
    spdlog::info("Loaded camera database with {} entries", _entries.size());
    return true;
}

bool CameraDatabase::isLoaded() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _loaded;
}

std::optional<CameraDBEntry> CameraDatabase::lookup(const image_metadata::camera_info_t &camera_info) const
{
    std::lock_guard<std::mutex> lock(_mutex);

    if (!_loaded)
    {
        return std::nullopt;
    }

    std::string make = toLower(camera_info.make);
    std::string model_name = toLower(camera_info.model);
    std::string lens_model = toLower(camera_info.lens_model);

    // Priority 1: Exact match (make + model + lens_model + dimensions)
    for (const auto &entry : _entries)
    {
        if (toLower(entry.make) == make && toLower(entry.model) == model_name &&
            toLower(entry.lens_model) == lens_model && entry.sensor_width_px == camera_info.width_px &&
            entry.sensor_height_px == camera_info.height_px)
        {
            return entry;
        }
    }

    // Priority 2: make + model + dimensions (ignore lens_model)
    for (const auto &entry : _entries)
    {
        if (toLower(entry.make) == make && toLower(entry.model) == model_name &&
            entry.sensor_width_px == camera_info.width_px && entry.sensor_height_px == camera_info.height_px)
        {
            return entry;
        }
    }

    // Priority 3: make + model only (for different resolutions/crops)
    for (const auto &entry : _entries)
    {
        if (toLower(entry.make) == make && toLower(entry.model) == model_name)
        {
            return entry;
        }
    }

    return std::nullopt;
}

void applyDatabaseEntry(const CameraDBEntry &entry, const image_metadata::camera_info_t &camera_info,
                        CameraModel &model)
{
    model.radial_distortion = entry.radial_distortion;
    model.tangential_distortion = entry.tangential_distortion;

    Eigen::Vector2d center(camera_info.width_px / 2.0, camera_info.height_px / 2.0);

    // Scale principal point offset if sensor dimensions differ
    if (entry.sensor_width_px != camera_info.width_px || entry.sensor_height_px != camera_info.height_px)
    {
        double scale = static_cast<double>(camera_info.width_px) / entry.sensor_width_px;
        model.principle_point = center + entry.principal_point_offset * scale;
    }
    else
    {
        model.principle_point = center + entry.principal_point_offset;
    }
}

} // namespace opencalibration
