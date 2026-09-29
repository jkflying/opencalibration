#include <opencalibration/extract/camera_database.hpp>

#include <spdlog/spdlog.h>

#define RAPIDJSON_HAS_STDSTRING 1
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include <opencalibration/io/json_fields.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <tuple>

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

    static CameraKey of(const opencalibration::CameraDBEntry &e)
    {
        return {e.make, e.model, e.lens_model, e.sensor_width_px, e.sensor_height_px};
    }

    bool operator<(const CameraKey &other) const
    {
        return std::tie(make, model, lens_model, width_px, height_px) <
               std::tie(other.make, other.model, other.lens_model, other.width_px, other.height_px);
    }
    bool operator==(const CameraKey &other) const
    {
        return std::tie(make, model, lens_model, width_px, height_px) ==
               std::tie(other.make, other.model, other.lens_model, other.width_px, other.height_px);
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

        const CameraKey key = CameraKey::of(entry);
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

struct ParsedDatabase
{
    std::vector<opencalibration::CameraDBEntry> entries;
    std::map<CameraKey, std::string> notes;
};

std::optional<ParsedDatabase> parseDatabase(const std::string &path)
{
    std::ifstream file(path);
    if (!file.is_open())
        return std::nullopt;

    std::stringstream buf;
    buf << file.rdbuf();

    rapidjson::Document doc;
    doc.Parse(buf.str().c_str());
    if (doc.HasParseError())
    {
        spdlog::warn("Failed to parse camera database {}: error at offset {}", path, doc.GetErrorOffset());
        return std::nullopt;
    }

    const auto *cameras = opencalibration::findJsonMember(doc, "cameras");
    if (!cameras || !cameras->IsArray())
    {
        spdlog::warn("Camera database {} has invalid structure", path);
        return std::nullopt;
    }
    int version = 0;
    if (!opencalibration::readJsonField(doc, "version", version) || version != 1)
    {
        spdlog::warn("Unsupported camera database version in {}", path);
        return std::nullopt;
    }

    using opencalibration::readJsonArrayField;
    using opencalibration::readJsonField;
    ParsedDatabase db;
    for (const auto &cam : cameras->GetArray())
    {
        if (!cam.IsObject())
            continue;

        opencalibration::CameraDBEntry entry;
        readJsonField(cam, "make", entry.make);
        readJsonField(cam, "model", entry.model);
        readJsonField(cam, "lens_model", entry.lens_model);
        readJsonField(cam, "sensor_width_px", entry.sensor_width_px);
        readJsonField(cam, "sensor_height_px", entry.sensor_height_px);
        readJsonArrayField(cam, "radial_distortion", entry.radial_distortion);
        readJsonArrayField(cam, "tangential_distortion", entry.tangential_distortion);
        readJsonArrayField(cam, "principal_point_offset", entry.principal_point_offset);
        readJsonField(cam, "focal_length_pixels", entry.focal_length_pixels);

        std::string notes;
        readJsonField(cam, "notes", notes);
        if (!notes.empty())
            db.notes[CameraKey::of(entry)] = notes;

        db.entries.push_back(std::move(entry));
    }
    return db;
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
    static const std::string path =
        std::filesystem::exists(CAMERA_DATABASE_PATH) ? CAMERA_DATABASE_PATH : CAMERA_DATABASE_INSTALL_PATH;
    return path;
}

bool CameraDatabase::load(const std::string &path)
{
    std::lock_guard<std::mutex> lock(_mutex);

    if (_loaded)
    {
        return true;
    }

    auto db = parseDatabase(path);
    if (!db)
    {
        spdlog::warn("Camera database not loaded from: {}", path);
        return false;
    }
    _entries = std::move(db->entries);

    _loaded = true;
    spdlog::info("Loaded camera database with {} entries", _entries.size());
    return true;
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
    const double scale =
        entry.sensor_width_px > 0 ? static_cast<double>(camera_info.width_px) / entry.sensor_width_px : 1.0;
    model.principle_point = center + entry.principal_point_offset * scale;

    // Apply focal_length_pixels ONLY if EXIF didn't provide valid value
    if (!std::isnan(entry.focal_length_pixels) &&
        (std::isnan(model.focal_length_pixels) || model.focal_length_pixels <= 0))
    {
        model.focal_length_pixels = entry.focal_length_pixels * scale;
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

    auto parsed = parseDatabase(database_path);
    std::vector<CameraDBEntry> db_entries = parsed ? std::move(parsed->entries) : std::vector<CameraDBEntry>{};
    std::map<CameraKey, std::string> notes_map = parsed ? std::move(parsed->notes) : std::map<CameraKey, std::string>{};

    for (const auto &[model_id, new_entry] : unique_models)
    {
        const CameraKey key = CameraKey::of(new_entry);

        bool found = false;
        for (auto &existing : db_entries)
        {
            if (CameraKey::of(existing) == key)
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
