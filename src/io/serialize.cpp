#include <opencalibration/io/serialize.hpp>

#include "base64.h"

#define RAPIDJSON_WRITE_DEFAULT_FLAGS kWriteNanAndInfFlag
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

namespace
{

template <size_t N> std::string bitset_to_bytes(const std::bitset<N> &bs)
{
    std::string result;
    result.resize((N + 7) >> 3, '\0');
    for (int j = 0; j < int(N); j++)
        result[j >> 3] |= (bs[j] << (j & 7));
    return result;
}

} // namespace
namespace opencalibration
{

template <> class Serializer<MeasurementGraph>
{
  public:
    static std::string to_json(const MeasurementGraph &graph)
    {
        rapidjson::StringBuffer buffer;

        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);

        writer.SetFormatOptions(rapidjson::PrettyFormatOptions::kFormatSingleLineArray);
        writer.StartObject();

        writer.Key("version");
        writer.Int64(1);

        writer.Key("nodes");
        writer.StartObject();
        // sort nodes by id to make repeatable with unordered map
        std::vector<size_t> node_ids;
        node_ids.reserve(graph._nodes.size());
        for (const auto &kv : graph._nodes)
        {
            node_ids.push_back(kv.first);
        }
        std::sort(node_ids.begin(), node_ids.end());
        for (size_t node_id : node_ids)
        {
            std::string node_id_str = std::to_string(node_id);
            const auto &node = graph._nodes.find(node_id)->second;
            writer.Key(node_id_str.c_str(), node_id_str.size());
            writer.StartObject();

            writer.Key("path");
            writer.String(node.payload.path.c_str());

            writer.Key("model");
            writer.StartObject();

            writer.Key("dimensions");
            writer.StartArray();
            writer.Uint64(node.payload.model.pixels_cols);
            writer.Uint64(node.payload.model.pixels_rows);
            writer.EndArray();

            writer.Key("focal_length");
            writer.Double(node.payload.model.focal_length_pixels);

            writer.Key("principal");
            writer.StartArray();
            writer.Double(node.payload.model.principle_point[0]);
            writer.Double(node.payload.model.principle_point[1]);
            writer.EndArray();

            writer.Key("projection");
            switch (node.payload.model.projection_type)
            {
            case CameraModel::ProjectionType::PLANAR:
                writer.String("planar");
                break;
            case CameraModel::ProjectionType::UNKNOWN:
                writer.String("UNKNOWN");
                break;
            }

            writer.EndObject();

            writer.Key("edges");

            writer.StartArray();
            for (const auto &edge : node.getEdges())
            {
                std::string edge_id = std::to_string(edge);
                writer.String(edge_id.c_str(), edge_id.size());
            }
            writer.EndArray();

            writer.Key("metadata");
            writer.StartObject();

            writer.Key("dimensions");
            writer.StartArray();
            writer.Uint64(node.payload.metadata.width_px);
            writer.Uint64(node.payload.metadata.height_px);
            writer.EndArray();

            writer.Key("focal_length_px");
            writer.Double(node.payload.metadata.focal_length_px);

            writer.Key("principal");
            writer.StartArray();
            writer.Double(node.payload.metadata.principal_point_px[0]);
            writer.Double(node.payload.metadata.principal_point_px[1]);
            writer.EndArray();

            writer.Key("latitude");
            writer.Double(node.payload.metadata.latitude);

            writer.Key("longitude");
            writer.Double(node.payload.metadata.longitude);

            writer.Key("altitude");
            writer.Double(node.payload.metadata.altitude);

            writer.Key("relative_altitude");
            writer.Double(node.payload.metadata.relativeAltitude);

            writer.Key("roll");
            writer.Double(node.payload.metadata.rollDegree);

            writer.Key("pitch");
            writer.Double(node.payload.metadata.pitchDegree);

            writer.Key("yaw");
            writer.Double(node.payload.metadata.yawDegree);

            writer.Key("accuracy_xy");
            writer.Double(node.payload.metadata.accuracyXY);

            writer.Key("accuracy_z");
            writer.Double(node.payload.metadata.accuracyZ);

            writer.Key("datum");
            writer.String(node.payload.metadata.datum.c_str());

            writer.Key("timestamp");
            writer.String(node.payload.metadata.timestamp.c_str());

            writer.Key("datestamp");
            writer.String(node.payload.metadata.datestamp.c_str());

            writer.EndObject();

            writer.Key("features");
            writer.StartArray();

            for (const auto &feature : node.payload.features)
            {
                writer.StartObject();

                writer.Key("location");
                writer.StartArray();
                writer.Double(feature.location.x());
                writer.Double(feature.location.y());
                writer.EndArray();

                writer.Key("strength");
                writer.Double(feature.strength);

                writer.Key("descriptor");
                std::string descriptor = bitset_to_bytes(feature.descriptor);
                std::string base64_descriptor;
                base64_descriptor.resize(Base64encode_len(descriptor.size()));
                int actual_size =
                    Base64encode(const_cast<char *>(base64_descriptor.c_str()), descriptor.c_str(), descriptor.size());
                writer.String(base64_descriptor.c_str(), actual_size - 1);

                writer.EndObject();
            }
            writer.EndArray();

            writer.EndObject();
        }
        writer.EndObject();

        writer.Key("edges");
        writer.StartObject();
        std::vector<size_t> edge_ids;
        edge_ids.reserve(graph._edges.size());
        for (const auto &kv : graph._edges)
        {
            edge_ids.push_back(kv.first);
        }
        std::sort(edge_ids.begin(), edge_ids.end());
        for (size_t edge_id : edge_ids)
        {
            std::string edge_id_str = std::to_string(edge_id);
            const auto &edge = graph._edges.find(edge_id)->second;

            writer.Key(edge_id_str.c_str(), edge_id_str.size());
            writer.StartObject();

            writer.Key("source");
            std::string source_id = std::to_string(edge.getSource());
            writer.String(source_id.c_str(), source_id.size());

            writer.Key("dest");
            std::string dest_id = std::to_string(edge.getDest());
            writer.String(dest_id.c_str(), dest_id.size());

            writer.Key("keypoints");

            writer.StartArray();
            for (const auto &pair : edge.payload.inlier_matches)
            {
                writer.StartArray();

                writer.StartArray();
                writer.Double(pair.pixel_1.x());
                writer.Double(pair.pixel_1.y());
                writer.EndArray();

                writer.StartArray();
                writer.Double(pair.pixel_2.x());
                writer.Double(pair.pixel_2.y());
                writer.EndArray();

                writer.EndArray();
            }
            writer.EndArray();

            writer.Key("relation");
            writer.StartArray();
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    writer.Double(edge.payload.ransac_relation(i, j));
                }
            }
            writer.EndArray();

            writer.Key("relation_type");
            switch (edge.payload.relationType)
            {
                case camera_relations::RelationType::HOMOGRAPHY:
                    writer.String("homography");
                    break;
                case camera_relations::RelationType::UNKNOWN:
                    writer.String("UNKNOWN");
                    break;
            }

            writer.Key("relative_rotation");
            writer.StartArray();
            for (int i = 0; i < 4; i++)
            {
                writer.Double(edge.payload.relative_rotation.coeffs()(i));
            }
            writer.EndArray();

            writer.Key("relative_translation");
            writer.StartArray();
            for (int i = 0; i < 3; i++)
            {
                writer.Double(edge.payload.relative_translation(i));
            }
            writer.EndArray();

            writer.EndObject();
        }
        writer.EndObject();

        writer.EndObject();

        return buffer.GetString();
    }
};
std::string serialize(const MeasurementGraph &graph)
{
    return Serializer<MeasurementGraph>::to_json(graph);
}
} // namespace opencalibration
