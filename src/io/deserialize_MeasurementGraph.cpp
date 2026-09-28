#include <opencalibration/io/deserialize.hpp>

#include "base64.h"

#include <opencv2/imgcodecs.hpp>

#define RAPIDJSON_HAS_STDSTRING 1
#define RAPIDJSON_PARSE_DEFAULT_FLAGS (kParseFullPrecisionFlag | kParseNanAndInfFlag)

#include <opencalibration/io/cv_raster_conversion.hpp>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/reader.h>
#include <rapidjson/stringbuffer.h>

namespace
{
class BufferedIStream
{
  public:
    typedef char Ch;
    explicit BufferedIStream(std::istream &in) : _sb(in.rdbuf()), _buf(1 << 16)
    {
    }
    Ch Peek()
    {
        if (_pos == _len)
        {
            _consumed += _len;
            _len = static_cast<size_t>(_sb->sgetn(_buf.data(), static_cast<std::streamsize>(_buf.size())));
            _pos = 0;
        }
        return _pos < _len ? _buf[_pos] : '\0';
    }
    Ch Take()
    {
        const Ch c = Peek();
        if (_pos < _len)
            _pos++;
        return c;
    }
    [[nodiscard]] size_t Tell() const
    {
        return _consumed + _pos;
    }
    Ch *PutBegin()
    {
        return nullptr;
    }
    void Put(Ch)
    {
    }
    void Flush()
    {
    }
    size_t PutEnd(Ch *)
    {
        return 0;
    }

  private:
    std::streambuf *_sb;
    std::vector<char> _buf;
    size_t _pos = 0, _len = 0, _consumed = 0;
};

template <size_t N> std::bitset<N> bitset_from_bytes(const std::string &buf)
{
    assert(buf.size() == ((N + 7) >> 3));
    std::bitset<N> result;
    for (int j = 0; j < int(N); j++)
        result[j] = ((buf[j >> 3] >> (j & 7)) & 1);
    return result;
}
} // namespace

namespace opencalibration
{
template <> class Deserializer<MeasurementGraph>
{
  public:
    template <typename Value> static void readFeatures(const Value &features, std::vector<feature_2d> &out)
    {
        std::string descriptor;
        out.reserve(features.Size());
        for (const auto &feat : features.GetArray())
        {
            feature_2d f;
            const char *base64_descriptor = feat.GetObject()["descriptor"].GetString();
            descriptor.resize(Base64decode_len(base64_descriptor), '\0');
            int actual_size = Base64decode(const_cast<char *>(descriptor.c_str()), base64_descriptor);
            descriptor.resize(actual_size);
            f.descriptor = bitset_from_bytes<feature_2d::DESCRIPTOR_BITS>(descriptor);

            f.location.x() = feat.GetObject()["location"].GetArray()[0].GetDouble();
            f.location.y() = feat.GetObject()["location"].GetArray()[1].GetDouble();

            f.strength = feat.GetObject()["strength"].GetDouble();

            out.push_back(f);
        }
    }

    struct FeaturesHandler : rapidjson::BaseReaderHandler<rapidjson::UTF8<>, FeaturesHandler>
    {
        MeasurementGraph &graph;
        std::vector<feature_2d> *out = nullptr;
        feature_2d f;
        std::string key, descriptor;
        int depth = 0, loc_idx = 0;

        explicit FeaturesHandler(MeasurementGraph &g) : graph(g)
        {
        }
        bool StartObject()
        {
            if (++depth == 3)
                f = {};
            return true;
        }
        bool EndObject(rapidjson::SizeType)
        {
            if (depth-- == 3 && out != nullptr)
                out->push_back(f);
            return true;
        }
        bool StartArray()
        {
            if (++depth == 4)
                loc_idx = 0;
            return true;
        }
        bool EndArray(rapidjson::SizeType)
        {
            if (--depth == 1)
                out = nullptr;
            return true;
        }
        bool Key(const char *s, rapidjson::SizeType len, bool)
        {
            if (depth == 1)
            {
                auto *node = graph.getNode(std::strtoull(s, nullptr, 10));
                out = node != nullptr ? &node->payload.features : nullptr;
                if (out != nullptr)
                    out->clear();
            }
            else
            {
                key.assign(s, len);
            }
            return true;
        }
        bool String(const char *s, rapidjson::SizeType, bool)
        {
            if (depth == 3 && key == "descriptor")
            {
                descriptor.resize(Base64decode_len(s), '\0');
                descriptor.resize(Base64decode(descriptor.data(), s));
                f.descriptor = bitset_from_bytes<feature_2d::DESCRIPTOR_BITS>(descriptor);
            }
            return true;
        }
        bool Double(double d)
        {
            if (depth == 4 && loc_idx < 2)
                f.location[loc_idx++] = d;
            else if (depth == 3 && key == "strength")
                f.strength = d;
            return true;
        }
        bool Int(int i)
        {
            return Double(i);
        }
        bool Uint(unsigned i)
        {
            return Double(i);
        }
        bool Int64(int64_t i)
        {
            return Double(static_cast<double>(i));
        }
        bool Uint64(uint64_t i)
        {
            return Double(static_cast<double>(i));
        }
    };

    static bool features_from_json(std::istream &in, MeasurementGraph &graph)
    {
        BufferedIStream stream(in);
        FeaturesHandler handler(graph);
        rapidjson::Reader reader;
        return !reader.Parse<rapidjson::kParseFullPrecisionFlag | rapidjson::kParseNanAndInfFlag>(stream, handler)
                    .IsError();
    }

    template <typename Input> static bool from_json(Input &input, MeasurementGraph &graph)
    {
        typedef rapidjson::GenericDocument<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<>,
                                           rapidjson::MemoryPoolAllocator<>>
            DocumentType;
        char valueBuffer[4096];
        char parseBuffer[1024];
        rapidjson::MemoryPoolAllocator<> valueAllocator(valueBuffer, sizeof(valueBuffer));
        rapidjson::MemoryPoolAllocator<> parseAllocator(parseBuffer, sizeof(parseBuffer));

        ankerl::unordered_dense::map<size_t, std::shared_ptr<CameraModel>> camera_models;

        DocumentType d(&valueAllocator, sizeof(parseBuffer), &parseAllocator);
        if constexpr (std::is_same_v<Input, const std::string>)
        {
            d.Parse(input);
        }
        else
        {
            BufferedIStream stream(input);
            d.ParseStream(stream);
        }

        bool parsed = false;
        char *end = nullptr;
        if (d.IsObject())
        {
            const auto &base = d.GetObject();
            if (base.HasMember("version") && base["version"].IsInt64() &&
                (base["version"].GetInt64() == 1 || base["version"].GetInt64() == 2))
            {
                const auto &nodes = base["nodes"].GetObject();
                for (const auto &node_member : nodes)
                {
                    size_t node_id = std::strtoull(node_member.name.GetString(), &end, 10);
                    image img;
                    img.path = node_member.value.GetObject()["path"].GetString();

                    const auto &position = node_member.value.GetObject()["position"].GetArray();
                    for (int i = 0; i < 3; i++)
                    {
                        img.position[i] = position[i].GetDouble();
                    }

                    if (node_member.value.GetObject().HasMember("gps_position"))
                    {
                        const auto &gps_position = node_member.value.GetObject()["gps_position"].GetArray();
                        for (int i = 0; i < 3; i++)
                        {
                            img.gps_position[i] = gps_position[i].GetDouble();
                        }
                    }

                    const auto &orientation = node_member.value.GetObject()["orientation"].GetArray();
                    for (int i = 0; i < 4; i++)
                    {
                        img.orientation.coeffs()[i] = orientation[i].GetDouble();
                    }

                    const char *base64_thumbnail = node_member.value.GetObject()["thumbnail"].GetString();
                    std::vector<uchar> thumbnail;
                    thumbnail.resize(Base64decode_len(base64_thumbnail), '\0');
                    int actual_size = Base64decode(reinterpret_cast<char *>(thumbnail.data()), base64_thumbnail);
                    thumbnail.resize(actual_size);
                    cv::Mat cvThumbnail;
                    cv::imdecode(thumbnail, cv::IMREAD_COLOR, &cvThumbnail);
                    img.thumbnail = RasterToRGB(cvToRaster(cvThumbnail));

                    const auto &model = node_member.value.GetObject()["model"].GetObject();

                    // TODO: dedup this by looking at the other image models for model ID

                    size_t id = model["id"].GetInt64();
                    auto iter = camera_models.find(id);
                    if (iter == camera_models.end())
                    {
                        img.model = std::make_shared<CameraModel>();
                        img.model->id = id;
                        img.model->pixels_cols = model["dimensions"].GetArray()[0].GetInt64();
                        img.model->pixels_rows = model["dimensions"].GetArray()[1].GetInt64();
                        img.model->focal_length_pixels = model["focal_length"].GetDouble();
                        img.model->principle_point[0] = model["principal"].GetArray()[0].GetDouble();
                        img.model->principle_point[1] = model["principal"].GetArray()[1].GetDouble();
                        img.model->radial_distortion[0] = model["radial_distortion"].GetArray()[0].GetDouble();
                        img.model->radial_distortion[1] = model["radial_distortion"].GetArray()[1].GetDouble();
                        img.model->radial_distortion[2] = model["radial_distortion"].GetArray()[2].GetDouble();
                        img.model->tangential_distortion[0] = model["tangential_distortion"].GetArray()[0].GetDouble();
                        img.model->tangential_distortion[1] = model["tangential_distortion"].GetArray()[1].GetDouble();
                        img.model->projection_type = "planar" == std::string(model["projection"].GetString())
                                                         ? ProjectionType::PLANAR
                                                         : ProjectionType::UNKNOWN;
                        camera_models.emplace(id, img.model);
                    }
                    else
                    {
                        img.model = iter->second;
                    }

                    MeasurementGraph::Node node(std::move(img));

                    const auto &edge_ids = node_member.value.GetObject()["edges"].GetArray();
                    for (const auto &edge_id : edge_ids)
                    {
                        node._edges.insert(std::strtoull(edge_id.GetString(), &end, 10));
                    }

                    {
                        const auto &metadata = node_member.value.GetObject()["metadata"].GetObject();
                        {
                            const auto &camera_info_j = metadata["camera_info"].GetObject();
                            auto &camera_info = node.payload.metadata.camera_info;
                            camera_info.width_px = camera_info_j["dimensions"].GetArray()[0].GetInt64();
                            camera_info.height_px = camera_info_j["dimensions"].GetArray()[1].GetInt64();

                            camera_info.focal_length_px = camera_info_j["focal_length_px"].GetDouble();

                            camera_info.principal_point_px[0] = camera_info_j["principal"].GetArray()[0].GetDouble();
                            camera_info.principal_point_px[1] = camera_info_j["principal"].GetArray()[1].GetDouble();

                            camera_info.make = camera_info_j["make"].GetString();
                            camera_info.model = camera_info_j["model"].GetString();
                            camera_info.serial_no = camera_info_j["serial_no"].GetString();
                            camera_info.lens_make = camera_info_j["lens_make"].GetString();
                            camera_info.lens_model = camera_info_j["lens_model"].GetString();
                        }

                        {
                            const auto &capture_info_j = metadata["capture_info"].GetObject();
                            auto &capture_info = node.payload.metadata.capture_info;
                            capture_info.latitude = capture_info_j["latitude"].GetDouble();
                            capture_info.longitude = capture_info_j["longitude"].GetDouble();
                            capture_info.altitude = capture_info_j["altitude"].GetDouble();
                            capture_info.relativeAltitude = capture_info_j["relative_altitude"].GetDouble();

                            capture_info.rollDegree = capture_info_j["roll"].GetDouble();
                            capture_info.pitchDegree = capture_info_j["pitch"].GetDouble();
                            capture_info.yawDegree = capture_info_j["yaw"].GetDouble();

                            capture_info.accuracyXY = capture_info_j["accuracy_xy"].GetDouble();
                            capture_info.accuracyZ = capture_info_j["accuracy_z"].GetDouble();

                            capture_info.datum = capture_info_j["datum"].GetString();
                            capture_info.timestamp = capture_info_j["timestamp"].GetString();
                            capture_info.datestamp = capture_info_j["datestamp"].GetString();
                        }
                    }

                    const auto &node_obj = node_member.value.GetObject();
                    if (node_obj.HasMember("features"))
                    {
                        readFeatures(node_obj["features"], node.payload.features);
                    }

                    if (node_obj.HasMember("num_sparse_features"))
                    {
                        node.payload.num_sparse_features = node_obj["num_sparse_features"].GetUint64();
                    }
                    else
                    {
                        node.payload.num_sparse_features = node.payload.features.size();
                    }

                    graph._nodes.emplace(node_id, std::move(node));
                }
                const auto &edges = base["edges"].GetObject();
                for (const auto &edge_member : edges)
                {
                    size_t edge_id = std::strtoull(edge_member.name.GetString(), &end, 10);
                    size_t source = std::strtoull(edge_member.value.GetObject()["source"].GetString(), &end, 10);
                    size_t dest = std::strtoull(edge_member.value.GetObject()["dest"].GetString(), &end, 10);

                    graph._edge_id_from_nodes_lookup.emplace(MeasurementGraph::SourceDestIndex{source, dest}, edge_id);

                    camera_relations relations;
                    const auto &matches = edge_member.value.GetObject()["matches"].GetArray();
                    relations.matches.reserve(matches.Size());
                    for (const auto &m : matches)
                    {
                        feature_match fm;
                        fm.feature_index_1 = m[0].GetInt64();
                        fm.feature_index_2 = m[1].GetInt64();
                        fm.distance = m[2].GetDouble();

                        relations.matches.push_back(fm);
                    }

                    const auto &inlier_matches = edge_member.value.GetObject()["inlier_matches"].GetArray();
                    relations.inlier_matches.reserve(inlier_matches.Size());
                    for (const auto &kp : inlier_matches)
                    {
                        feature_match_denormalized fmd;
                        fmd.pixel_1[0] = kp[0].GetArray()[0].GetDouble();
                        fmd.pixel_1[1] = kp[0].GetArray()[1].GetDouble();
                        fmd.pixel_2[0] = kp[1].GetArray()[0].GetDouble();
                        fmd.pixel_2[1] = kp[1].GetArray()[1].GetDouble();
                        fmd.feature_index_1 = kp[2].GetInt64();
                        fmd.feature_index_2 = kp[3].GetInt64();
                        fmd.match_index = kp[4].GetInt64();
                        relations.inlier_matches.push_back(fmd);
                    }
                    const auto &relation = edge_member.value.GetObject()["relation"].GetArray();
                    for (int i = 0; i < 3; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            relations.ransac_relation(i, j) = relation[i * 3 + j].GetDouble();
                        }
                    }

                    std::string rel_type = edge_member.value.GetObject()["relation_type"].GetString();
                    if (rel_type == "homography")
                    {
                        relations.relationType = camera_relations::RelationType::HOMOGRAPHY;
                    }
                    else if (rel_type == "fundamental_matrix")
                    {
                        relations.relationType = camera_relations::RelationType::FUNDAMENTAL_MATRIX;
                    }
                    else if (rel_type == "essential_matrix")
                    {
                        relations.relationType = camera_relations::RelationType::ESSENTIAL_MATRIX;
                    }
                    else
                    {
                        relations.relationType = camera_relations::RelationType::UNKNOWN;
                    }

                    const auto &rel_pose = edge_member.value.GetObject()["relative_pose"].GetArray();
                    for (size_t i = 0; i < rel_pose.Size(); i++)
                    {
                        relations.relative_poses[i].score = rel_pose[i].GetObject()["score"].GetInt();
                        const auto &rel_ori = rel_pose[i].GetObject()["orientation"].GetArray();
                        for (int j = 0; j < 4; j++)
                        {
                            relations.relative_poses[i].orientation.coeffs()(j) = rel_ori[j].GetDouble();
                        }

                        const auto &rel_pos = rel_pose[i].GetObject()["position"].GetArray();
                        for (int j = 0; j < 3; j++)
                        {
                            relations.relative_poses[i].position(j) = rel_pos[j].GetDouble();
                        }
                    }

                    MeasurementGraph::Edge edge(std::move(relations), source, dest);
                    graph._edges.emplace(edge_id, std::move(edge));
                }
                parsed = true;
            }
        }
        return parsed;
    }
};
bool deserialize(const std::string &json, MeasurementGraph &graph)
{
    return Deserializer<MeasurementGraph>::from_json(json, graph);
}

bool deserialize(std::istream &json, MeasurementGraph &graph)
{
    return Deserializer<MeasurementGraph>::from_json(json, graph);
}

bool deserializeFeatures(std::istream &json, MeasurementGraph &graph)
{
    return Deserializer<MeasurementGraph>::features_from_json(json, graph);
}
} // namespace opencalibration
