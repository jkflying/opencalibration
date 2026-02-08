#include <opencalibration/extract/camera_database.hpp>

#include <spdlog/spdlog.h>

#define RAPIDJSON_HAS_STDSTRING 1
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace
{
std::string toLower(const std::string &s)
{
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) { return std::tolower(c); });
    return result;
}

struct CameraKey
{
    std::string make;
    std::string model;
    std::string lens_model;
    size_t width_px;
    size_t height_px;

    bool operator<(const CameraKey &other) const
    {
        if (make != other.make)
            return make < other.make;
        if (model != other.model)
            return model < other.model;
        if (lens_model != other.lens_model)
            return lens_model < other.lens_model;
        if (width_px != other.width_px)
            return width_px < other.width_px;
        return height_px < other.height_px;
    }
};

opencalibration::CameraDBEntry extractEntry(const opencalibration::image &img)
{
    opencalibration::CameraDBEntry entry;
    entry.make = img.metadata.camera_info.make;
    entry.model = img.metadata.camera_info.model;
    entry.lens_model = img.metadata.camera_info.lens_model;
    entry.sensor_width_px = img.model->pixels_cols;
    entry.sensor_height_px = img.model->pixels_rows;
    entry.radial_distortion = img.model->radial_distortion;
    entry.tangential_distortion = img.model->tangential_distortion;

    Eigen::Vector2d center(img.model->pixels_cols / 2.0, img.model->pixels_rows / 2.0);
    entry.principal_point_offset = img.model->principle_point - center;

    if (img.model->focal_length_pixels > 0)
    {
        entry.focal_length_pixels = img.model->focal_length_pixels;
    }

    return entry;
}

void writeDatabase(const std::string &path, const std::vector<opencalibration::CameraDBEntry> &entries,
                   const std::map<CameraKey, std::string> &notes_map)
{
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    writer.SetIndent(' ', 2);
    writer.SetFormatOptions(rapidjson::kFormatSingleLineArray);

    writer.StartObject();
    writer.Key("version");
    writer.Int(1);
    writer.Key("cameras");
    writer.StartArray();

    for (const auto &entry : entries)
    {
        writer.StartObject();

        writer.Key("make");
        writer.String(entry.make);
        writer.Key("model");
        writer.String(entry.model);
        writer.Key("lens_model");
        writer.String(entry.lens_model);
        writer.Key("sensor_width_px");
        writer.Uint64(entry.sensor_width_px);
        writer.Key("sensor_height_px");
        writer.Uint64(entry.sensor_height_px);

        writer.Key("radial_distortion");
        writer.StartArray();
        for (int i = 0; i < 3; ++i)
            writer.Double(entry.radial_distortion[i]);
        writer.EndArray();

        writer.Key("tangential_distortion");
        writer.StartArray();
        for (int i = 0; i < 2; ++i)
            writer.Double(entry.tangential_distortion[i]);
        writer.EndArray();

        writer.Key("principal_point_offset");
        writer.StartArray();
        for (int i = 0; i < 2; ++i)
            writer.Double(entry.principal_point_offset[i]);
        writer.EndArray();

        if (!std::isnan(entry.focal_length_pixels))
        {
            writer.Key("focal_length_pixels");
            writer.Double(entry.focal_length_pixels);
        }

        CameraKey key{entry.make, entry.model, entry.lens_model, entry.sensor_width_px, entry.sensor_height_px};
        auto notes_it = notes_map.find(key);
        if (notes_it != notes_map.end())
        {
            writer.Key("notes");
            writer.String(notes_it->second);
        }

        writer.EndObject();
    }

    writer.EndArray();
    writer.EndObject();

    std::ofstream out(path);
    if (!out.is_open())
    {
        spdlog::error("Failed to open {} for writing", path);
        return;
    }
    out << buffer.GetString() << "\n";
    out.close();
    spdlog::info("Wrote camera database to {}", path);
}

std::vector<opencalibration::CameraDBEntry> loadDatabaseEntries(const std::string &path,
                                                                std::map<CameraKey, std::string> &notes_map)
{
    std::vector<opencalibration::CameraDBEntry> entries;

    std::ifstream file(path);
    if (!file.is_open())
    {
        spdlog::info("No existing database at {}, will create new", path);
        return entries;
    }

    std::stringstream buf;
    buf << file.rdbuf();
    std::string json = buf.str();

    rapidjson::Document doc;
    doc.Parse(json.c_str());

    if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("cameras"))
    {
        spdlog::warn("Failed to parse existing database at {}", path);
        return entries;
    }

    const auto &cameras = doc["cameras"].GetArray();
    for (const auto &cam : cameras)
    {
        opencalibration::CameraDBEntry entry;
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
                entry.radial_distortion[i] = rd[i].GetDouble();
        }
        if (cam.HasMember("tangential_distortion"))
        {
            const auto &td = cam["tangential_distortion"].GetArray();
            for (size_t i = 0; i < std::min(td.Size(), rapidjson::SizeType(2)); ++i)
                entry.tangential_distortion[i] = td[i].GetDouble();
        }
        if (cam.HasMember("principal_point_offset"))
        {
            const auto &pp = cam["principal_point_offset"].GetArray();
            for (size_t i = 0; i < std::min(pp.Size(), rapidjson::SizeType(2)); ++i)
                entry.principal_point_offset[i] = pp[i].GetDouble();
        }
        if (cam.HasMember("focal_length_pixels"))
        {
            entry.focal_length_pixels = cam["focal_length_pixels"].GetDouble();
        }

        CameraKey key{entry.make, entry.model, entry.lens_model, entry.sensor_width_px, entry.sensor_height_px};
        if (cam.HasMember("notes"))
            notes_map[key] = cam["notes"].GetString();

        entries.push_back(std::move(entry));
    }
    spdlog::info("Loaded existing database with {} entries", entries.size());

    return entries;
}

} // namespace

namespace opencalibration
{

CameraDatabase &CameraDatabase::instance()
{
    static CameraDatabase db;
    return db;
}

const std::string &CameraDatabase::defaultPath()
{
    static const std::string path = CAMERA_DATABASE_PATH;
    return path;
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

        if (cam.HasMember("focal_length_pixels"))
        {
            entry.focal_length_pixels = cam["focal_length_pixels"].GetDouble();
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

    // Apply focal_length_pixels ONLY if EXIF didn't provide valid value
    if (!std::isnan(entry.focal_length_pixels) &&
        (std::isnan(model.focal_length_pixels) || model.focal_length_pixels <= 0))
    {
        model.focal_length_pixels = entry.focal_length_pixels;
        spdlog::debug("Applied database focal length: {} pixels", model.focal_length_pixels);
    }
}

bool updateDatabaseFromGraph(const MeasurementGraph &graph, const std::string &database_path, const std::string &notes)
{
    std::map<size_t, CameraDBEntry> unique_models;
    for (auto it = graph.cnodebegin(); it != graph.cnodeend(); ++it)
    {
        const auto &img = it->second.payload;
        if (!img.model)
            continue;

        size_t model_id = img.model->id;
        if (unique_models.find(model_id) != unique_models.end())
            continue;

        unique_models[model_id] = extractEntry(img);
    }

    if (unique_models.empty())
    {
        spdlog::error("No camera models found in graph");
        return false;
    }

    spdlog::info("Found {} unique camera model(s)", unique_models.size());

    std::map<CameraKey, std::string> notes_map;
    std::vector<CameraDBEntry> db_entries = loadDatabaseEntries(database_path, notes_map);

    for (const auto &[model_id, new_entry] : unique_models)
    {
        CameraKey key{new_entry.make, new_entry.model, new_entry.lens_model, new_entry.sensor_width_px,
                      new_entry.sensor_height_px};

        bool found = false;
        for (auto &existing : db_entries)
        {
            CameraKey existing_key{existing.make, existing.model, existing.lens_model, existing.sensor_width_px,
                                   existing.sensor_height_px};
            if (!(existing_key < key) && !(key < existing_key))
            {
                existing.radial_distortion = new_entry.radial_distortion;
                existing.tangential_distortion = new_entry.tangential_distortion;
                existing.principal_point_offset = new_entry.principal_point_offset;
                existing.focal_length_pixels = new_entry.focal_length_pixels;
                spdlog::info("Updated existing entry: {} {}", key.make, key.model);
                found = true;
                break;
            }
        }

        if (!found)
        {
            db_entries.push_back(new_entry);
            spdlog::info("Added new entry: {} {}", key.make, key.model);
        }

        if (!notes.empty())
        {
            notes_map[key] = notes;
        }
    }

    writeDatabase(database_path, db_entries, notes_map);
    return true;
}

} // namespace opencalibration
