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

#include <zstd.h>

#include <filesystem>
#include <fstream>
#include <limits>

namespace opencalibration
{

namespace
{

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
        writer.Key("brdf_coeff");
        writer.Double(params.brdf_coeff);
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
            readJsonField(image, "brdf_coeff", params.brdf_coeff);
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

bool saveMetadata(const CheckpointData &data, const std::filesystem::path &path)
{
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

    writer.StartObject();

    writer.Key("version");
    writer.Int(2);

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

    std::ofstream out(path);
    if (!out.is_open())
    {
        spdlog::error("Failed to open metadata.json for writing");
        return false;
    }
    out << buffer.GetString();
    return true;
}

bool loadMetadata(CheckpointData &data, const std::filesystem::path &path, size_t &surface_count)
{
    std::ifstream in(path);
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

    int version = 0;
    if (!readJsonField(doc, "version", version) || version < 1 || version > 2)
    {
        spdlog::error("Unsupported checkpoint version");
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

bool savePointCloud(const point_cloud &cloud, const std::filesystem::path &filepath)
{
    std::ofstream out(filepath);
    if (!out.is_open())
    {
        spdlog::error("Failed to open {} for writing", filepath.string());
        return false;
    }

    out.precision(std::numeric_limits<double>::max_digits10);
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

class ZstdFileBuf : public std::streambuf
{
  public:
    explicit ZstdFileBuf(const std::filesystem::path &path)
        : _file(path, std::ios::binary), _cctx(ZSTD_createCCtx()), _out(ZSTD_CStreamOutSize())
    {
        ZSTD_CCtx_setParameter(_cctx, ZSTD_c_compressionLevel, 3);
        ZSTD_CCtx_setParameter(_cctx, ZSTD_c_nbWorkers, 2);
    }
    ~ZstdFileBuf() override
    {
        ZSTD_freeCCtx(_cctx);
    }
    bool close()
    {
        const bool ok = _ok && compress(nullptr, 0, ZSTD_e_end);
        _file.close();
        return ok && !_file.fail();
    }

  protected:
    std::streamsize xsputn(const char *s, std::streamsize n) override
    {
        _ok = _ok && compress(s, n, ZSTD_e_continue);
        return _ok ? n : 0;
    }
    int_type overflow(int_type ch) override
    {
        if (traits_type::eq_int_type(ch, traits_type::eof()))
            return traits_type::not_eof(ch);
        const char c = traits_type::to_char_type(ch);
        return xsputn(&c, 1) == 1 ? ch : traits_type::eof();
    }

  private:
    bool compress(const char *data, size_t size, ZSTD_EndDirective mode)
    {
        ZSTD_inBuffer in{data, size, 0};
        size_t remaining = 0;
        do
        {
            ZSTD_outBuffer out{_out.data(), _out.size(), 0};
            remaining = ZSTD_compressStream2(_cctx, &out, &in, mode);
            if (ZSTD_isError(remaining))
                return false;
            _file.write(_out.data(), out.pos);
        } while (mode == ZSTD_e_end ? remaining != 0 : in.pos != in.size);
        return _file.good();
    }

    std::ofstream _file;
    ZSTD_CCtx *_cctx;
    std::vector<char> _out;
    bool _ok = true;
};

template <typename Write> bool writeCompressed(const std::filesystem::path &path, Write &&write)
{
    ZstdFileBuf buf(path);
    std::ostream out(&buf);
    const bool written = write(out);
    return buf.close() && written;
}

class ZstdReadBuf : public std::streambuf
{
  public:
    explicit ZstdReadBuf(const std::filesystem::path &path)
        : _file(path, std::ios::binary), _dctx(ZSTD_createDCtx()), _in(ZSTD_DStreamInSize()),
          _out(ZSTD_DStreamOutSize())
    {
    }
    ~ZstdReadBuf() override
    {
        ZSTD_freeDCtx(_dctx);
    }

  protected:
    int_type underflow() override
    {
        while (true)
        {
            if (_input.pos == _input.size && !_output_full)
            {
                _file.read(_in.data(), static_cast<std::streamsize>(_in.size()));
                if (_file.gcount() == 0)
                    return traits_type::eof();
                _input = {_in.data(), static_cast<size_t>(_file.gcount()), 0};
            }
            ZSTD_outBuffer output{_out.data(), _out.size(), 0};
            if (ZSTD_isError(ZSTD_decompressStream(_dctx, &output, &_input)))
                return traits_type::eof();
            _output_full = output.pos == output.size;
            if (output.pos > 0)
            {
                setg(_out.data(), _out.data(), _out.data() + output.pos);
                return traits_type::to_int_type(_out[0]);
            }
        }
    }

  private:
    std::ifstream _file;
    ZSTD_DCtx *_dctx;
    std::vector<char> _in, _out;
    ZSTD_inBuffer _input{nullptr, 0, 0};
    bool _output_full = false;
};

template <typename Read> bool readJson(const std::filesystem::path &path, Read &&read)
{
    std::filesystem::path zst = path;
    zst += ".zst";
    if (std::filesystem::exists(zst))
    {
        ZstdReadBuf buf(zst);
        std::istream in(&buf);
        return read(in);
    }
    std::ifstream in(path, std::ios::binary);
    return in.is_open() && read(in);
}

std::filesystem::path stageFile(const std::filesystem::path &dir, const std::string &stage, const std::string &name)
{
    return dir / (stage.empty() ? name : stage + "_" + name);
}

std::string featuresFingerprint(const MeasurementGraph &graph)
{
    uint64_t fingerprint = 0;
    for (auto it = graph.cnodebegin(); it != graph.cnodeend(); ++it)
    {
        fingerprint += (it->first + 0x9e3779b97f4a7c15ull) * (it->second.payload.features.size() + 1);
    }
    return std::to_string(graph.size_nodes()) + " " + std::to_string(fingerprint);
}

std::string readFile(const std::filesystem::path &path)
{
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool saveFeatures(const MeasurementGraph &graph, const std::filesystem::path &dir)
{
    const std::string fingerprint = featuresFingerprint(graph);
    if (std::filesystem::exists(dir / "features.json.zst") && readFile(dir / "features.fingerprint") == fingerprint)
    {
        return true;
    }

    if (!writeCompressed(dir / "features.json.zst.tmp",
                         [&](std::ostream &out) { return serializeFeatures(graph, out); }))
    {
        spdlog::error("Failed to write features.json.zst");
        return false;
    }
    std::filesystem::rename(dir / "features.json.zst.tmp", dir / "features.json.zst");
    std::ofstream(dir / "features.fingerprint") << fingerprint;
    return true;
}

} // namespace

std::vector<CheckpointStage> listCheckpointStages(const std::string &checkpoint_dir)
{
    std::vector<CheckpointStage> stages;
    std::ifstream in(std::filesystem::path(checkpoint_dir) / "checkpoints.txt");
    std::string line;
    while (std::getline(in, line))
    {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos)
            continue;
        stages.push_back({line.substr(0, tab),
                          stringToPipelineState(line.substr(tab + 1)).value_or(PipelineState::INITIAL_PROCESSING)});
    }
    return stages;
}

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

    const size_t index = listCheckpointStages(checkpoint_dir).size();
    const std::string stage = (index < 10 ? "0" : "") + std::to_string(index) + "_" + pipelineStateToString(data.state);

    if (!saveMetadata(data, stageFile(dir, stage, "metadata.json")) || !saveFeatures(data.graph, dir))
    {
        return false;
    }

    if (!writeCompressed(stageFile(dir, stage, "graph.json.zst"),
                         [&](std::ostream &out) { return serialize(data.graph, out, false); }))
    {
        spdlog::error("Failed to serialize graph");
        return false;
    }

    for (size_t i = 0; i < data.surfaces.size(); i++)
    {
        const auto &surface = data.surfaces[i];

        if (surface.mesh.size_nodes() > 0)
        {
            std::string mesh_filename = "surface_" + std::to_string(i) + ".ply";
            std::ofstream mesh_out(stageFile(dir, stage, mesh_filename));
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
            if (!savePointCloud(surface.cloud[j], stageFile(dir, stage, cloud_filename)))
            {
                return false;
            }
        }

        std::string cloud_count_filename = "surface_" + std::to_string(i) + "_cloudcount.txt";
        std::ofstream count_out(stageFile(dir, stage, cloud_count_filename));
        if (count_out.is_open())
        {
            count_out << surface.cloud.size();
        }
    }

    std::ofstream(dir / "checkpoints.txt", std::ios::app) << stage << "\t" << pipelineStateToString(data.state) << "\n";

    spdlog::info("Checkpoint {} saved to {}", stage, checkpoint_dir);
    return true;
}

bool loadCheckpoint(const std::string &checkpoint_dir, CheckpointData &data, std::string stage)
{
    std::filesystem::path dir(checkpoint_dir);

    if (!std::filesystem::exists(dir))
    {
        spdlog::error("Checkpoint directory does not exist: {}", checkpoint_dir);
        return false;
    }

    const auto stages = listCheckpointStages(checkpoint_dir);
    if (stage.empty() && !stages.empty())
    {
        stage = stages.back().name;
    }

    size_t surface_count = 0;
    if (!loadMetadata(data, stageFile(dir, stage, "metadata.json"), surface_count))
    {
        return false;
    }

    if (!readJson(stageFile(dir, stage, "graph.json"), [&](std::istream &in) { return deserialize(in, data.graph); }))
    {
        spdlog::error("Failed to deserialize graph");
        return false;
    }

    if (!stage.empty() &&
        !readJson(dir / "features.json", [&](std::istream &in) { return deserializeFeatures(in, data.graph); }))
    {
        spdlog::error("Failed to deserialize features");
        return false;
    }

    data.surfaces.clear();
    data.surfaces.resize(surface_count);

    for (size_t i = 0; i < surface_count; i++)
    {
        auto &surface = data.surfaces[i];

        std::string mesh_filename = "surface_" + std::to_string(i) + ".ply";
        std::filesystem::path mesh_path = stageFile(dir, stage, mesh_filename);
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
        std::filesystem::path count_path = stageFile(dir, stage, cloud_count_filename);
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
            std::filesystem::path cloud_path = stageFile(dir, stage, cloud_filename);
            if (std::filesystem::exists(cloud_path))
            {
                if (!loadPointCloud(surface.cloud[j], cloud_path))
                {
                    spdlog::warn("Failed to load point cloud {} for surface {}", j, i);
                }
            }
        }
    }

    spdlog::info("Checkpoint {} loaded from {}", stage, checkpoint_dir);
    return true;
}

} // namespace opencalibration
