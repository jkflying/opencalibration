#include <opencalibration/io/serialize.hpp>

#include "base64.h"

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

std::string serialize(const MeasurementGraph &graph)
{
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);

    writer.SetFormatOptions(rapidjson::PrettyFormatOptions::kFormatSingleLineArray);
    //     writer.SetMaxDecimalPlaces(11);
    writer.StartObject();

    writer.Key("version");
    writer.Int64(1);

    writer.Key("nodes");
    writer.StartObject();
    for (auto iter = graph.nodebegin(); iter != graph.nodeend(); ++iter)
    {
        std::string node_id = std::to_string(iter->first);
        writer.Key(node_id.c_str(), node_id.size());
        writer.StartObject();

        writer.Key("path");
        writer.String(iter->second.payload.path.c_str());

        writer.Key("model");
        writer.StartObject();

        writer.Key("dimensions");
        writer.StartArray();
        writer.Uint64(iter->second.payload.model.pixels_cols);
        writer.Uint64(iter->second.payload.model.pixels_rows);
        writer.EndArray();

        writer.Key("focal_length");
        writer.Double(iter->second.payload.model.focal_length_pixels);

        writer.Key("principal");
        writer.StartArray();
        writer.Double(iter->second.payload.model.principle_point[0]);
        writer.Double(iter->second.payload.model.principle_point[1]);
        writer.EndArray();

        writer.Key("projection");
        switch (iter->second.payload.model.projection_type)
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
        for (const auto &edge : iter->second.getEdges())
        {
            std::string edge_id = std::to_string(edge);
            writer.String(edge_id.c_str(), edge_id.size());
        }
        writer.EndArray();

        // TODO: metadata

        writer.Key("features");
        writer.StartArray();

        for (const auto &feature : iter->second.payload.features)
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
    for (auto iter = graph.edgebegin(); iter != graph.edgeend(); ++iter)
    {
        std::string edge_id = std::to_string(iter->first);
        writer.Key(edge_id.c_str(), edge_id.size());
        writer.StartObject();

        writer.Key("source");
        std::string source_id = std::to_string(iter->second.getSource());
        writer.String(source_id.c_str(), source_id.size());

        writer.Key("dest");
        std::string dest_id = std::to_string(iter->second.getDest());
        writer.String(dest_id.c_str(), dest_id.size());

        writer.Key("keypoints");

        writer.StartArray();
        for (const auto &pair : iter->second.payload.inlier_matches)
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

        writer.EndObject();
    }
    writer.EndObject();

    writer.EndObject();

    return buffer.GetString();
}
} // namespace opencalibration
