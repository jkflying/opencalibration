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
                    img.model.pixels_cols = model["dimensions"].GetArray()[0].GetInt64();
                    img.model.pixels_rows = model["dimensions"].GetArray()[1].GetInt64();
                    img.model.focal_length_pixels = model["focal_length"].GetDouble();
                    img.model.principle_point[0] = model["principal"].GetArray()[0].GetDouble();
                    img.model.principle_point[1] = model["principal"].GetArray()[1].GetDouble();
                    img.model.projection_type = "planar" == std::string(model["projection"].GetString())
                                                    ? ProjectionType::PLANAR
                                                    : ProjectionType::UNKNOWN;

                    MeasurementGraph::Node node(std::move(img));

                    const auto &edge_ids = niter->value.GetObject()["edges"].GetArray();
                    for (const auto &edge_id : edge_ids)
                    {
                        node._edges.push_back(std::strtoull(edge_id.GetString(), &end, 10));
                    }

                    {
                        const auto &metadata = niter->value.GetObject()["metadata"].GetObject();
                        node.payload.metadata.width_px = metadata["dimensions"].GetArray()[0].GetInt64();
                        node.payload.metadata.height_px = metadata["dimensions"].GetArray()[1].GetInt64();

                        node.payload.metadata.focal_length_px = metadata["focal_length_px"].GetDouble();

                        node.payload.metadata.principal_point_px[0] = metadata["principal"].GetArray()[0].GetDouble();
                        node.payload.metadata.principal_point_px[1] = metadata["principal"].GetArray()[1].GetDouble();

                        node.payload.metadata.latitude = metadata["latitude"].GetDouble();
                        node.payload.metadata.longitude = metadata["longitude"].GetDouble();
                        node.payload.metadata.altitude = metadata["altitude"].GetDouble();
                        node.payload.metadata.relativeAltitude = metadata["relative_altitude"].GetDouble();

                        node.payload.metadata.rollDegree = metadata["roll"].GetDouble();
                        node.payload.metadata.pitchDegree = metadata["pitch"].GetDouble();
                        node.payload.metadata.yawDegree = metadata["yaw"].GetDouble();

                        node.payload.metadata.accuracyXY = metadata["accuracy_xy"].GetDouble();
                        node.payload.metadata.accuracyZ = metadata["accuracy_z"].GetDouble();

                        node.payload.metadata.datum = metadata["datum"].GetString();
                        node.payload.metadata.timestamp = metadata["timestamp"].GetString();
                        node.payload.metadata.datestamp = metadata["datestamp"].GetString();
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
                    const auto &keypoints = eiter->value.GetObject()["keypoints"].GetArray();
                    relations.inlier_matches.reserve(keypoints.Size());
                    for (const auto &kp : keypoints)
                    {
                        feature_match_denormalized fmd;
                        fmd.pixel_1[0] = kp[0].GetArray()[0].GetDouble();
                        fmd.pixel_1[1] = kp[0].GetArray()[1].GetDouble();
                        fmd.pixel_2[0] = kp[1].GetArray()[0].GetDouble();
                        fmd.pixel_2[1] = kp[1].GetArray()[1].GetDouble();
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

                    const auto &rel_rot = eiter->value.GetObject()["relative_rotation"].GetArray();
                    for (int i = 0; i < 4; i++)
                    {
                        relations.relative_rotation.coeffs()(i) = rel_rot[i].GetDouble();
                    }

                    const auto &rel_trans = eiter->value.GetObject()["relative_translation"].GetArray();
                    for (int i = 0; i < 3; i++)
                    {
                        relations.relative_translation(i) = rel_trans[i].GetDouble();
                    }

                    const auto &rel_rot2 = eiter->value.GetObject()["relative_rotation2"].GetArray();
                    for (int i = 0; i < 4; i++)
                    {
                        relations.relative_rotation2.coeffs()(i) = rel_rot2[i].GetDouble();
                    }

                    const auto &rel_trans2 = eiter->value.GetObject()["relative_translation2"].GetArray();
                    for (int i = 0; i < 3; i++)
                    {
                        relations.relative_translation2(i) = rel_trans2[i].GetDouble();
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
