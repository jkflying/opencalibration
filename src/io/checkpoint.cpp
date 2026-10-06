#include <opencalibration/io/checkpoint.hpp>

#include <opencalibration/io/deserialize.hpp>
#include <opencalibration/io/serialize.hpp>

#include <spdlog/spdlog.h>

#define RAPIDJSON_HAS_STDSTRING 1
#define RAPIDJSON_WRITE_DEFAULT_FLAGS kWriteNanAndInfFlag
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <opencalibration/io/json_fields.hpp>

#include <sqlite3.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <map>
#include <sstream>

namespace opencalibration
{

namespace
{

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "project.db blobs are little-endian"
#endif

template <typename Writer, size_t N> void writeArray(Writer &writer, const std::array<double, N> &values)
{
    writer.StartArray();
    for (double v : values)
        writer.Double(v);
    writer.EndArray();
}

template <typename Writer> void writeColorBalance(Writer &writer, const orthomosaic::ColorBalanceResult &cb)
{
    writer.StartObject();
    writer.Key("success");
    writer.Bool(cb.success);
    writer.Key("final_cost");
    writer.Double(cb.final_cost);
    writer.Key("num_iterations");
    writer.Int(cb.num_iterations);
    writer.Key("horizontal_view_dir_log_cbrt_gain");
    writeArray(writer, cb.horizontal_view_dir_log_cbrt_gain);

    writer.Key("images");
    writer.StartArray();
    for (const auto &[id, params] : cb.per_image_params)
    {
        writer.StartObject();
        writer.Key("id");
        writer.Uint64(id);
        writer.Key("log_cbrt_exposure");
        writer.Double(params.log_cbrt_exposure);
        writer.Key("ab_offset");
        writeArray(writer, params.ab_offset);
        writer.Key("slope");
        writeArray(writer, params.slope);
        writer.EndObject();
    }
    writer.EndArray();

    writer.Key("models");
    writer.StartArray();
    for (const auto &[id, params] : cb.per_model_params)
    {
        writer.StartObject();
        writer.Key("id");
        writer.Uint(id);
        writer.Key("log_cbrt_falloff_coeffs");
        writeArray(writer, params.log_cbrt_falloff_coeffs);
        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();
}

void readColorBalance(const rapidjson::Value &value, orthomosaic::ColorBalanceResult &cb)
{
    readJsonField(value, "success", cb.success);
    readJsonField(value, "final_cost", cb.final_cost);
    readJsonField(value, "num_iterations", cb.num_iterations);
    readJsonArrayField(value, "horizontal_view_dir_log_cbrt_gain", cb.horizontal_view_dir_log_cbrt_gain);

    const auto *images = findJsonMember(value, "images");
    if (images && images->IsArray())
        for (const auto &image : images->GetArray())
        {
            uint64_t id = 0;
            if (!readJsonField(image, "id", id))
                continue;
            auto &params = cb.per_image_params[id];
            readJsonField(image, "log_cbrt_exposure", params.log_cbrt_exposure);
            readJsonArrayField(image, "ab_offset", params.ab_offset);
            readJsonArrayField(image, "slope", params.slope);
        }

    const auto *models = findJsonMember(value, "models");
    if (models && models->IsArray())
        for (const auto &model : models->GetArray())
        {
            unsigned id = 0;
            if (readJsonField(model, "id", id))
                readJsonArrayField(model, "log_cbrt_falloff_coeffs", cb.per_model_params[id].log_cbrt_falloff_coeffs);
        }
}

std::string metadataJson(const CheckpointData &data)
{
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

    writer.StartObject();
    writer.Key("version");
    writer.Int(3);
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
    if (!data.color_balance.per_image_params.empty())
    {
        writer.Key("color_balance");
        writeColorBalance(writer, data.color_balance);
    }
    writer.EndObject();
    return {buffer.GetString(), buffer.GetSize()};
}

bool parseMetadata(const std::string &json, CheckpointData &data, size_t &surface_count)
{
    rapidjson::Document doc;
    int version = 0;
    if (doc.Parse(json.c_str()).HasParseError() || !readJsonField(doc, "version", version) || version != 3)
    {
        spdlog::error("Unsupported checkpoint metadata");
        return false;
    }

    std::string state;
    if (readJsonField(doc, "state", state))
        data.state = stringToPipelineState(state).value_or(PipelineState::INITIAL_PROCESSING);
    readJsonField(doc, "state_run_count", data.state_run_count);
    readJsonField(doc, "origin_latitude", data.origin_latitude);
    readJsonField(doc, "origin_longitude", data.origin_longitude);
    readJsonField(doc, "surface_count", surface_count);
    if (const auto *color_balance = findJsonMember(doc, "color_balance"))
        readColorBalance(*color_balance, data.color_balance);
    return true;
}

template <typename T> void put(std::string &out, const T &value)
{
    out.append(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <typename T> bool get(const char *&p, const char *end, T &value)
{
    if (static_cast<size_t>(end - p) < sizeof(T))
        return false;
    std::memcpy(&value, p, sizeof(T));
    p += sizeof(T);
    return true;
}

constexpr uint32_t FEATURES_VERSION = 1;
constexpr uint32_t FEATURE_RECORD = 2 * sizeof(double) + sizeof(float) + Descriptor::WORDS * sizeof(uint64_t);

std::string encodeFeatures(const std::vector<feature_2d> &features)
{
    std::string out;
    out.reserve(8 + features.size() * FEATURE_RECORD);
    put(out, FEATURES_VERSION);
    put(out, FEATURE_RECORD);
    for (const auto &f : features)
    {
        put(out, f.location.x());
        put(out, f.location.y());
        put(out, f.strength);
        for (uint64_t word : f.descriptor.words)
            put(out, word);
    }
    return out;
}

std::vector<feature_2d> decodeFeatures(const char *p, size_t size)
{
    const char *end = p + size;
    uint32_t version = 0, record = 0;
    std::vector<feature_2d> features;
    if (!get(p, end, version) || !get(p, end, record) || version != FEATURES_VERSION || record != FEATURE_RECORD)
        return features;
    features.resize((end - p) / FEATURE_RECORD);
    for (auto &f : features)
    {
        get(p, end, f.location.x());
        get(p, end, f.location.y());
        get(p, end, f.strength);
        for (uint64_t &word : f.descriptor.words)
            get(p, end, word);
    }
    return features;
}

std::string encodeSurface(const surface_model &surface)
{
    std::ostringstream ply;
    serialize(surface.mesh, ply);
    const std::string mesh = ply.str();
    std::string out;
    put(out, static_cast<uint64_t>(mesh.size()));
    out += mesh;
    put(out, static_cast<uint64_t>(surface.cloud.size()));
    for (const auto &cloud : surface.cloud)
    {
        put(out, static_cast<uint64_t>(cloud.size()));
        for (const auto &point : cloud)
            for (int i = 0; i < 3; i++)
                put(out, point[i]);
    }
    return out;
}

bool decodeSurface(const std::string &blob, surface_model &surface)
{
    const char *p = blob.data(), *end = p + blob.size();
    uint64_t mesh_size = 0, clouds = 0;
    if (!get(p, end, mesh_size) || static_cast<uint64_t>(end - p) < mesh_size)
        return false;
    if (mesh_size > 0)
    {
        std::istringstream ply(std::string(p, mesh_size));
        if (!deserialize(ply, surface.mesh))
            return false;
    }
    p += mesh_size;
    if (!get(p, end, clouds))
        return false;
    surface.cloud.resize(clouds);
    for (auto &cloud : surface.cloud)
    {
        uint64_t points = 0;
        if (!get(p, end, points) || static_cast<uint64_t>(end - p) < points * 3 * sizeof(double))
            return false;
        cloud.resize(points);
        for (auto &point : cloud)
            for (int i = 0; i < 3; i++)
                get(p, end, point[i]);
    }
    return true;
}

std::string fileStamp(const std::string &path)
{
    std::error_code ec1, ec2;
    const auto size = std::filesystem::file_size(path, ec1);
    const auto time = std::filesystem::last_write_time(path, ec2);
    if (ec1 || ec2)
        return {};
    return std::to_string(size) + ":" + std::to_string(time.time_since_epoch().count());
}

constexpr const char *FEATURES_ONLY_STAMP = "";

class Statement
{
  public:
    Statement(sqlite3 *db, const char *sql)
    {
        if (sqlite3_prepare_v2(db, sql, -1, &_stmt, nullptr) != SQLITE_OK)
            spdlog::error("SQL prepare failed: {}", sqlite3_errmsg(db));
    }
    ~Statement()
    {
        sqlite3_finalize(_stmt);
    }
    Statement(const Statement &) = delete;
    Statement &operator=(const Statement &) = delete;

    Statement &reset()
    {
        sqlite3_reset(_stmt);
        sqlite3_clear_bindings(_stmt);
        return *this;
    }
    Statement &bind(int i, int64_t value)
    {
        sqlite3_bind_int64(_stmt, i, value);
        return *this;
    }
    Statement &bind(int i, const std::string &text)
    {
        sqlite3_bind_text(_stmt, i, text.data(), static_cast<int>(text.size()), SQLITE_TRANSIENT);
        return *this;
    }
    Statement &bindBlob(int i, const std::string &blob)
    {
        sqlite3_bind_blob64(_stmt, i, blob.data(), blob.size(), SQLITE_TRANSIENT);
        return *this;
    }

    bool nextRow()
    {
        return sqlite3_step(_stmt) == SQLITE_ROW;
    }
    bool run()
    {
        const bool ok = sqlite3_step(_stmt) == SQLITE_DONE;
        sqlite3_reset(_stmt);
        return ok;
    }

    int64_t integer(int column)
    {
        return sqlite3_column_int64(_stmt, column);
    }
    std::string text(int column)
    {
        const auto *data = static_cast<const char *>(sqlite3_column_blob(_stmt, column));
        return data ? std::string(data, sqlite3_column_bytes(_stmt, column)) : std::string();
    }

  private:
    sqlite3_stmt *_stmt = nullptr;
};

enum RowKind : int64_t
{
    NODE_ROW = 0,
    EDGE_ROW = 1,
    SURFACE_ROW = 2,
};

int64_t rowId(size_t id)
{
    return static_cast<int64_t>(id);
}

int64_t rowHash(const std::string &data)
{
    return static_cast<int64_t>(std::hash<std::string>{}(data));
}

} // namespace

std::shared_ptr<ProjectStore> ProjectStore::open(const std::string &dir)
{
    static std::mutex mutex;
    static std::map<std::string, std::weak_ptr<ProjectStore>> stores;

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string key = std::filesystem::weakly_canonical(dir, ec).string();

    std::lock_guard<std::mutex> lock(mutex);
    if (auto store = stores[key].lock())
        return store;

    sqlite3 *db = nullptr;
    const std::string path = (std::filesystem::path(dir) / "project.db").string();
    if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX, nullptr) !=
        SQLITE_OK)
    {
        spdlog::error("Failed to open {}: {}", path, sqlite3_errmsg(db));
        sqlite3_close(db);
        return nullptr;
    }

    std::shared_ptr<ProjectStore> store(new ProjectStore(db));
    if (!store->exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"
                     "CREATE TABLE IF NOT EXISTS images(path TEXT PRIMARY KEY, stamp TEXT NOT NULL, data TEXT NOT "
                     "NULL, num_features INTEGER NOT NULL, features BLOB NOT NULL);"
                     "CREATE TABLE IF NOT EXISTS stages(rev INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, state "
                     "TEXT NOT NULL, data TEXT NOT NULL);"
                     "CREATE TABLE IF NOT EXISTS rows(kind INTEGER, id INTEGER, rev INTEGER, hash INTEGER, data BLOB, "
                     "PRIMARY KEY(kind, id, rev)) WITHOUT ROWID;"))
        return nullptr;
    stores[key] = store;
    return store;
}

ProjectStore::ProjectStore(sqlite3 *db) : _db(db)
{
}

ProjectStore::~ProjectStore()
{
    sqlite3_close(_db);
}

bool ProjectStore::exec(const char *sql)
{
    char *error = nullptr;
    if (sqlite3_exec(_db, sql, nullptr, nullptr, &error) != SQLITE_OK)
    {
        spdlog::error("SQL failed: {}", error ? error : "?");
        sqlite3_free(error);
        return false;
    }
    return true;
}

bool ProjectStore::putImage(const std::string &path, const std::string &stamp, const image &img,
                            const std::vector<feature_2d> &features)
{
    MeasurementGraph single_node_graph;
    const size_t id = single_node_graph.addNode(image(img));
    const std::string node = serializeNode(single_node_graph, id), blob = encodeFeatures(features);

    std::lock_guard<std::recursive_mutex> lock(_mutex);
    return Statement(_db, "INSERT OR REPLACE INTO images VALUES(?, ?, ?, ?, ?)")
        .bind(1, path)
        .bind(2, stamp)
        .bind(3, node)
        .bind(4, static_cast<int64_t>(features.size()))
        .bindBlob(5, blob)
        .run();
}

std::vector<feature_2d> ProjectStore::readFeatures(const std::string &path)
{
    std::string blob;
    {
        const auto start = std::chrono::steady_clock::now();
        std::lock_guard<std::recursive_mutex> lock(_mutex);
        FeatureSet::loadStats().lock_wait_nanoseconds +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        Statement query(_db, "SELECT features FROM images WHERE path = ?");
        query.bind(1, path);
        if (!query.nextRow())
        {
            spdlog::error("No stored features for {}", path);
            return {};
        }
        blob = query.text(0);
    }
    return decodeFeatures(blob.data(), blob.size());
}

FeatureSet ProjectStore::storedFeatures(const std::string &path, size_t size)
{
    return FeatureSet::stored(size, [store = shared_from_this(), path] { return store->readFeatures(path); });
}

std::optional<image> ProjectStore::loadImage(const std::string &path)
{
    const std::string stamp = fileStamp(path);
    if (stamp.empty())
        return std::nullopt;

    std::string data;
    size_t num_features = 0;
    {
        std::lock_guard<std::recursive_mutex> lock(_mutex);
        Statement query(_db, "SELECT data, num_features FROM images WHERE path = ? AND stamp = ?");
        query.bind(1, path).bind(2, stamp);
        if (!query.nextRow())
            return std::nullopt;
        data = query.text(0);
        num_features = query.integer(1);
    }

    MeasurementGraph single_node_graph;
    if (!GraphRowReader().addNode(single_node_graph, 0, data))
        return std::nullopt;
    image img = std::move(single_node_graph.getNode(0)->payload);
    img.features = storedFeatures(path, num_features);
    return img;
}

bool ProjectStore::saveImage(image &img)
{
    std::vector<feature_2d> features = std::exchange(img.features, {}).load();
    if (!putImage(img.path, fileStamp(img.path), img, features))
    {
        img.features = std::move(features);
        return false;
    }
    img.features = storedFeatures(img.path, features.size());
    return true;
}

std::vector<CheckpointStage> ProjectStore::stages()
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    std::vector<CheckpointStage> stages;
    Statement query(_db, "SELECT name, state FROM stages ORDER BY rev");
    while (query.nextRow())
        stages.push_back(
            {query.text(0), stringToPipelineState(query.text(1)).value_or(PipelineState::INITIAL_PROCESSING)});
    return stages;
}

bool ProjectStore::save(const CheckpointData &data)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);
    if (!exec("BEGIN IMMEDIATE"))
        return false;

    int64_t rev = 1, index = 0;
    {
        Statement query(_db, "SELECT COALESCE(MAX(rev), 0) + 1, COUNT(*) FROM stages");
        if (query.nextRow())
        {
            rev = query.integer(0);
            index = query.integer(1);
        }
    }
    const std::string stage = (index < 10 ? "0" : "") + std::to_string(index) + "_" + pipelineStateToString(data.state);

    std::array<ankerl::unordered_dense::map<int64_t, int64_t>, 3> latest_row_hashes;
    {
        Statement query(_db, "SELECT r.kind, r.id, r.hash FROM rows r JOIN (SELECT kind, id, MAX(rev) m FROM rows "
                             "GROUP BY kind, id) l ON r.kind = l.kind AND r.id = l.id AND r.rev = l.m "
                             "WHERE r.hash IS NOT NULL");
        while (query.nextRow())
            latest_row_hashes[query.integer(0)][query.integer(1)] = query.integer(2);
    }

    bool ok = true;
    Statement insert(_db, "INSERT INTO rows VALUES(?, ?, ?, ?, ?)");
    auto writeRow = [&](RowKind kind, int64_t id, const std::string &row) {
        const int64_t hash = rowHash(row);
        auto it = latest_row_hashes[kind].find(id);
        const bool unchanged = it != latest_row_hashes[kind].end() && it->second == hash;
        if (it != latest_row_hashes[kind].end())
            latest_row_hashes[kind].erase(it);
        if (!unchanged)
            ok = insert.reset().bind(1, kind).bind(2, id).bind(3, rev).bind(4, hash).bindBlob(5, row).run() && ok;
    };

    Statement stored_image(_db, "SELECT 1 FROM images WHERE path = ?");
    for (auto it = data.graph.cnodebegin(); it != data.graph.cnodeend(); ++it)
    {
        writeRow(NODE_ROW, rowId(it->first), serializeNode(data.graph, it->first));
        const auto &img = it->second.payload;
        if (!stored_image.reset().bind(1, img.path).nextRow())
            ok = putImage(img.path, FEATURES_ONLY_STAMP, img, img.features.load()) && ok;
    }
    for (auto it = data.graph.cedgebegin(); it != data.graph.cedgeend(); ++it)
        writeRow(EDGE_ROW, rowId(it->first), serializeEdge(data.graph, it->first));
    for (size_t i = 0; i < data.surfaces.size(); i++)
        writeRow(SURFACE_ROW, static_cast<int64_t>(i), encodeSurface(data.surfaces[i]));

    const auto &rows_removed_since_latest = latest_row_hashes;
    for (int64_t kind = 0; kind < 3; kind++)
        for (const auto &[id, hash] : rows_removed_since_latest[kind])
            ok = insert.reset().bind(1, kind).bind(2, id).bind(3, rev).run() && ok;

    ok = Statement(_db, "INSERT INTO stages VALUES(?, ?, ?, ?)")
             .bind(1, rev)
             .bind(2, stage)
             .bind(3, pipelineStateToString(data.state))
             .bind(4, metadataJson(data))
             .run() &&
         ok;

    if (!ok || !exec("COMMIT"))
    {
        exec("ROLLBACK");
        spdlog::error("Failed to save checkpoint {}", stage);
        return false;
    }
    spdlog::info("Checkpoint {} saved", stage);
    return true;
}

bool ProjectStore::load(CheckpointData &data, const std::string &stage)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    int64_t rev = 0;
    std::string name;
    size_t surface_count = 0;
    {
        Statement query(_db, stage.empty() ? "SELECT rev, name, data FROM stages ORDER BY rev DESC LIMIT 1"
                                           : "SELECT rev, name, data FROM stages WHERE name = ?");
        if (!stage.empty())
            query.bind(1, stage);
        if (!query.nextRow())
        {
            spdlog::error("No checkpoint stage '{}'", stage);
            return false;
        }
        rev = query.integer(0);
        name = query.text(1);
        if (!parseMetadata(query.text(2), data, surface_count))
            return false;
    }

    Statement rows(_db, "SELECT r.id, r.data FROM rows r JOIN (SELECT id, MAX(rev) m FROM rows WHERE kind = ?1 AND "
                        "rev <= ?2 GROUP BY id) l ON r.id = l.id AND r.rev = l.m WHERE r.kind = ?1 AND r.data IS NOT "
                        "NULL");
    GraphRowReader reader;
    bool ok = true;

    rows.bind(1, NODE_ROW).bind(2, rev);
    while (rows.nextRow())
        ok = reader.addNode(data.graph, static_cast<size_t>(rows.integer(0)), rows.text(1)) && ok;

    rows.reset().bind(1, EDGE_ROW).bind(2, rev);
    while (rows.nextRow())
        ok = reader.addEdge(data.graph, static_cast<size_t>(rows.integer(0)), rows.text(1)) && ok;

    data.surfaces.clear();
    data.surfaces.resize(surface_count);
    rows.reset().bind(1, SURFACE_ROW).bind(2, rev);
    while (rows.nextRow())
    {
        const auto i = static_cast<size_t>(rows.integer(0));
        ok = i < surface_count && decodeSurface(rows.text(1), data.surfaces[i]) && ok;
    }

    Statement num_features(_db, "SELECT num_features FROM images WHERE path = ?");
    for (auto it = data.graph.nodebegin(); it != data.graph.nodeend(); ++it)
    {
        auto &img = it->second.payload;
        num_features.reset().bind(1, img.path);
        if (num_features.nextRow())
            img.features = storedFeatures(img.path, num_features.integer(0));
        else
            ok = false;
    }

    if (!ok)
    {
        spdlog::error("Checkpoint {} is incomplete", name);
        return false;
    }
    spdlog::info("Checkpoint {} loaded", name);
    return true;
}

std::vector<CheckpointStage> listCheckpointStages(const std::string &checkpoint_dir)
{
    if (!std::filesystem::exists(std::filesystem::path(checkpoint_dir) / "project.db"))
        return {};
    auto store = ProjectStore::open(checkpoint_dir);
    return store ? store->stages() : std::vector<CheckpointStage>{};
}

bool saveCheckpoint(const CheckpointData &data, const std::string &checkpoint_dir)
{
    auto store = ProjectStore::open(checkpoint_dir);
    return store && store->save(data);
}

bool loadCheckpoint(const std::string &checkpoint_dir, CheckpointData &data, std::string stage)
{
    if (!std::filesystem::exists(std::filesystem::path(checkpoint_dir) / "project.db"))
    {
        spdlog::error("No checkpoint in {}", checkpoint_dir);
        return false;
    }
    auto store = ProjectStore::open(checkpoint_dir);
    return store && store->load(data, stage);
}

} // namespace opencalibration
