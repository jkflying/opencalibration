#include <opencalibration/io/deserialize.hpp>

#include "base64.h"

#define RAPIDJSON_PARSE_DEFAULT_FLAGS (kParseFullPrecisionFlag | kParseNanAndInfFlag)

#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

namespace
{
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
    static bool from_json(const std::string &json, MeasurementGraph &graph)
    {
        typedef rapidjson::GenericDocument<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<>,
                                           rapidjson::MemoryPoolAllocator<>>
            DocumentType;
        char valueBuffer[4096];
        char parseBuffer[1024];
        rapidjson::MemoryPoolAllocator<> valueAllocator(valueBuffer, sizeof(valueBuffer));
        rapidjson::MemoryPoolAllocator<> parseAllocator(parseBuffer, sizeof(parseBuffer));

        std::unordered_map<size_t, std::shared_ptr<CameraModel>> camera_models;

        DocumentType d(&valueAllocator, sizeof(parseBuffer), &parseAllocator);
        d.Parse(json.c_str());

        bool parsed = false;
        char *end = nullptr;
        if (d.IsObject())
        {
            const auto &base = d.GetObject();
            if (base.HasMember("version") && base["version"].IsInt64() && base["version"].GetInt64() == 1)
            {
                const auto &nodes = base["nodes"].GetObject();
                for (auto niter = nodes.begin(); niter != nodes.end(); ++niter)
                {
                    size_t node_id = std::strtoull(niter->name.GetString(), &end, 10);
                    image img;
                    img.path = niter->value.GetObject()["path"].GetString();

                    const auto &position = niter->value.GetObject()["position"].GetArray();
                    for (int i = 0; i < 3; i++)
                    {
                        img.position[i] = position[i].GetDouble();
                    }

                    const auto &orientation = niter->value.GetObject()["orientation"].GetArray();
                    for (int i = 0; i < 4; i++)
                    {
                        img.orientation.coeffs()[i] = orientation[i].GetDouble();
                    }

                    const auto &model = niter->value.GetObject()["model"].GetObject();

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

                    const auto &edge_ids = niter->value.GetObject()["edges"].GetArray();
                    for (const auto &edge_id : edge_ids)
                    {
                        node._edges.push_back(std::strtoull(edge_id.GetString(), &end, 10));
                    }

                    {
                        const auto &metadata = niter->value.GetObject()["metadata"].GetObject();
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

                    const auto &features = niter->value.GetObject()["features"].GetArray();
                    std::string descriptor;
                    for (auto fiter = features.begin(); fiter != features.end(); ++fiter)
                    {
                        feature_2d f;
                        const char *base64_descriptor = fiter->GetObject()["descriptor"].GetString();
                        descriptor.resize(Base64decode_len(base64_descriptor), '\0');
                        int actual_size = Base64decode(const_cast<char *>(descriptor.c_str()), base64_descriptor);
                        descriptor.resize(actual_size);
                        f.descriptor = bitset_from_bytes<feature_2d::DESCRIPTOR_BITS>(descriptor);

                        f.location.x() = fiter->GetObject()["location"].GetArray()[0].GetDouble();
                        f.location.y() = fiter->GetObject()["location"].GetArray()[1].GetDouble();

                        f.strength = fiter->GetObject()["strength"].GetDouble();

                        node.payload.features.push_back(f);
                    }

                    graph._nodes.emplace(node_id, std::move(node));
                }
                const auto &edges = base["edges"].GetObject();
                for (auto eiter = edges.begin(); eiter != edges.end(); ++eiter)
                {
                    size_t edge_id = std::strtoull(eiter->name.GetString(), &end, 10);
                    size_t source = std::strtoull(eiter->value.GetObject()["source"].GetString(), &end, 10);
                    size_t dest = std::strtoull(eiter->value.GetObject()["dest"].GetString(), &end, 10);

                    camera_relations relations;
                    const auto &matches = eiter->value.GetObject()["matches"].GetArray();
                    relations.matches.reserve(matches.Size());
                    for (const auto &m : matches)
                    {
                        feature_match fm;
                        fm.feature_index_1 = m[0].GetInt64();
                        fm.feature_index_2 = m[1].GetInt64();
                        fm.distance = m[2].GetDouble();

                        relations.matches.push_back(fm);
                    }

                    const auto &inlier_matches = eiter->value.GetObject()["inlier_matches"].GetArray();
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
                        relations.inlier_matches.push_back(fmd);
                    }
                    const auto &relation = eiter->value.GetObject()["relation"].GetArray();
                    for (int i = 0; i < 3; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            relations.ransac_relation(i, j) = relation[i * 3 + j].GetDouble();
                        }
                    }

                    std::string rel_type = eiter->value.GetObject()["relation_type"].GetString();
                    if (rel_type == "homography")
                    {
                        relations.relationType = camera_relations::RelationType::HOMOGRAPHY;
                    }
                    else
                    {
                        relations.relationType = camera_relations::RelationType::UNKNOWN;
                    }

                    const auto &rel_pose = eiter->value.GetObject()["relative_pose"].GetArray();
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
} // namespace opencalibration
