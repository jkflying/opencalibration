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
    static std::string visualizeGeoJson(const MeasurementGraph &graph,
                                        std::function<Eigen::Vector3d(const Eigen::Vector3d &)> toGlobalCoordinates)
    {
        // sort nodes by id to make repeatable with unordered map
        std::vector<size_t> node_ids;
        node_ids.reserve(graph._nodes.size());
        for (const auto &kv : graph._nodes)
        {
            node_ids.push_back(kv.first);
        }
        std::sort(node_ids.begin(), node_ids.end());

        // sort edges by id to make repeatable with unordered map
        std::vector<size_t> edge_ids;
        edge_ids.reserve(graph._edges.size());
        for (const auto &kv : graph._edges)
        {
            edge_ids.push_back(kv.first);
        }
        std::sort(edge_ids.begin(), edge_ids.end());

        return visualizeGeoJson(graph, node_ids, edge_ids, toGlobalCoordinates);
    }

    static std::string visualizeGeoJson(const MeasurementGraph &graph, const std::vector<size_t> &node_ids,
                                        const std::vector<size_t> &edge_ids,
                                        std::function<Eigen::Vector3d(const Eigen::Vector3d &)> toGlobalCoordinates)
    {
        rapidjson::StringBuffer buffer;

        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);

        writer.SetFormatOptions(rapidjson::PrettyFormatOptions::kFormatSingleLineArray);
        writer.StartObject();
        writer.Key("type");
        writer.String("FeatureCollection");

        writer.Key("features");
        writer.StartArray();

        for (size_t node_id : node_ids)
        {
            writer.StartObject();
            writer.Key("type");
            writer.String("Feature");
            std::string node_id_str = std::to_string(node_id);
            const auto &node = graph._nodes.find(node_id)->second;

            writer.Key("geometry");
            writer.StartObject();
            {
                writer.Key("type");
                writer.String("Point");
                writer.Key("coordinates");
                writer.StartArray();
                Eigen::Vector3d wgs84 = toGlobalCoordinates(node.payload.position);
                for (int i = 0; i < 3; i++)
                {
                    writer.Double(wgs84[i]);
                }
                writer.EndArray();
            }
            writer.EndObject();

            writer.Key("properties");
            writer.StartObject();
            {
                writer.Key("path");
                writer.String(node.payload.path.c_str());
                writer.Key("id");
                writer.String(node_id_str.c_str(), node_id_str.size());
            }
            writer.EndObject();

            writer.EndObject();
        }

        for (size_t edge_id : edge_ids)
        {
            std::string edge_id_str = std::to_string(edge_id);
            const auto &edge = graph._edges.find(edge_id)->second;

            writer.StartObject();
            writer.Key("type");
            writer.String("Feature");

            writer.Key("geometry");
            writer.StartObject();
            {
                writer.Key("type");
                writer.String("LineString");
                writer.Key("coordinates");
                writer.StartArray();

                Eigen::Vector3d source_wgs84 =
                    toGlobalCoordinates(graph._nodes.find(edge.getSource())->second.payload.position);
                writer.StartArray();
                for (int i = 0; i < 3; i++)
                {
                    writer.Double(source_wgs84[i]);
                }
                writer.EndArray();

                Eigen::Vector3d dest_wgs84 =
                    toGlobalCoordinates(graph._nodes.find(edge.getDest())->second.payload.position);
                writer.StartArray();
                for (int i = 0; i < 3; i++)
                {
                    writer.Double(dest_wgs84[i]);
                }
                writer.EndArray();

                writer.EndArray();
            }

            writer.EndObject();

            writer.Key("properties");
            writer.StartObject();
            {
                writer.Key("id");
                writer.String(edge_id_str.c_str(), edge_id_str.size());

                writer.Key("source_id");
                std::string source_id = std::to_string(edge.getSource());
                writer.String(source_id.c_str(), source_id.size());

                writer.Key("dest_id");
                std::string dest_id = std::to_string(edge.getDest());
                writer.String(dest_id.c_str(), dest_id.size());

                writer.Key("inliers");
                writer.Int64(edge.payload.inlier_matches.size());
            }
            writer.EndObject();

            writer.EndObject();
        }

        writer.EndArray();

        writer.EndObject();

        return buffer.GetString();
    }

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
            {
                writer.Key("path");
                writer.String(node.payload.path.c_str());

                writer.Key("position");
                writer.StartArray();
                for (int i = 0; i < 3; i++)
                {
                    writer.Double(node.payload.position[i]);
                }
                writer.EndArray();

                writer.Key("orientation");
                writer.StartArray();
                for (int i = 0; i < 4; i++)
                {
                    writer.Double(node.payload.orientation.coeffs()[i]);
                }
                writer.EndArray();

                writer.Key("model");
                writer.StartObject();
                {
                    writer.Key("id");
                    writer.Int64(node.payload.model->id);

                    writer.Key("dimensions");
                    writer.StartArray();
                    {
                        writer.Uint64(node.payload.model->pixels_cols);
                        writer.Uint64(node.payload.model->pixels_rows);
                    }
                    writer.EndArray();

                    writer.Key("focal_length");
                    writer.Double(node.payload.model->focal_length_pixels);

                    writer.Key("principal");
                    writer.StartArray();
                    {
                        writer.Double(node.payload.model->principle_point[0]);
                        writer.Double(node.payload.model->principle_point[1]);
                    }
                    writer.EndArray();

                    writer.Key("radial_distortion");
                    writer.StartArray();
                    {
                        writer.Double(node.payload.model->radial_distortion[0]);
                        writer.Double(node.payload.model->radial_distortion[1]);
                        writer.Double(node.payload.model->radial_distortion[2]);
                    }
                    writer.EndArray();

                    writer.Key("tangential_distortion");
                    writer.StartArray();
                    {
                        writer.Double(node.payload.model->tangential_distortion[0]);
                        writer.Double(node.payload.model->tangential_distortion[1]);
                    }
                    writer.EndArray();

                    writer.Key("projection");
                    switch (node.payload.model->projection_type)
                    {
                    case ProjectionType::PLANAR:
                        writer.String("planar");
                        break;
                    case ProjectionType::UNKNOWN:
                        writer.String("UNKNOWN");
                        break;
                    }
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
                {
                    writer.Key("camera_info");
                    writer.StartObject();
                    {
                        const auto &camera_info = node.payload.metadata.camera_info;
                        writer.Key("dimensions");
                        writer.StartArray();
                        {
                            writer.Uint64(camera_info.width_px);
                            writer.Uint64(camera_info.height_px);
                        }
                        writer.EndArray();

                        writer.Key("focal_length_px");
                        writer.Double(camera_info.focal_length_px);

                        writer.Key("principal");
                        writer.StartArray();
                        {
                            writer.Double(camera_info.principal_point_px[0]);
                            writer.Double(camera_info.principal_point_px[1]);
                        }
                        writer.EndArray();

                        writer.Key("make");
                        writer.String(camera_info.make.c_str());

                        writer.Key("model");
                        writer.String(camera_info.model.c_str());

                        writer.Key("serial_no");
                        writer.String(camera_info.serial_no.c_str());

                        writer.Key("lens_make");
                        writer.String(camera_info.lens_make.c_str());

                        writer.Key("lens_model");
                        writer.String(camera_info.lens_model.c_str());
                    }
                    writer.EndObject();

                    writer.Key("capture_info");
                    writer.StartObject();
                    {
                        const auto &capture_info = node.payload.metadata.capture_info;
                        writer.Key("latitude");
                        writer.Double(capture_info.latitude);

                        writer.Key("longitude");
                        writer.Double(capture_info.longitude);

                        writer.Key("altitude");
                        writer.Double(capture_info.altitude);

                        writer.Key("relative_altitude");
                        writer.Double(capture_info.relativeAltitude);

                        writer.Key("roll");
                        writer.Double(capture_info.rollDegree);

                        writer.Key("pitch");
                        writer.Double(capture_info.pitchDegree);

                        writer.Key("yaw");
                        writer.Double(capture_info.yawDegree);

                        writer.Key("accuracy_xy");
                        writer.Double(capture_info.accuracyXY);

                        writer.Key("accuracy_z");
                        writer.Double(capture_info.accuracyZ);

                        writer.Key("datum");
                        writer.String(capture_info.datum.c_str());

                        writer.Key("timestamp");
                        writer.String(capture_info.timestamp.c_str());

                        writer.Key("datestamp");
                        writer.String(capture_info.datestamp.c_str());
                    }
                    writer.EndObject();
                }
                writer.EndObject();

                writer.Key("features");
                writer.StartArray();

                for (const auto &feature : node.payload.features)
                {
                    writer.StartObject();

                    writer.Key("location");
                    writer.StartArray();
                    {
                        writer.Double(feature.location.x());
                        writer.Double(feature.location.y());
                    }
                    writer.EndArray();

                    writer.Key("strength");
                    writer.Double(feature.strength);

                    writer.Key("descriptor");
                    std::string descriptor = bitset_to_bytes(feature.descriptor);
                    std::string base64_descriptor;
                    base64_descriptor.resize(Base64encode_len(descriptor.size()));
                    int actual_size = Base64encode(const_cast<char *>(base64_descriptor.c_str()), descriptor.c_str(),
                                                   descriptor.size());
                    writer.String(base64_descriptor.c_str(), actual_size - 1);

                    writer.EndObject();
                }
                writer.EndArray();
            }
            writer.EndObject();
        }
        writer.EndObject();

        writer.Key("edges");
        writer.StartObject();
        {
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
                {
                    writer.Key("source");
                    std::string source_id = std::to_string(edge.getSource());
                    writer.String(source_id.c_str(), source_id.size());

                    writer.Key("dest");
                    std::string dest_id = std::to_string(edge.getDest());
                    writer.String(dest_id.c_str(), dest_id.size());

                    writer.Key("matches");
                    writer.StartArray();
                    for (const auto &match : edge.payload.matches)
                    {
                        writer.StartArray();
                        writer.Int64(match.feature_index_1);
                        writer.Int64(match.feature_index_2);
                        writer.Double(match.distance);
                        writer.EndArray();
                    }
                    writer.EndArray();

                    writer.Key("inlier_matches");
                    writer.StartArray();
                    for (const auto &pair : edge.payload.inlier_matches)
                    {
                        writer.StartArray();

                        writer.StartArray();
                        {
                            writer.Double(pair.pixel_1.x());
                            writer.Double(pair.pixel_1.y());
                        }
                        writer.EndArray();

                        writer.StartArray();
                        {
                            writer.Double(pair.pixel_2.x());
                            writer.Double(pair.pixel_2.y());
                        }
                        writer.EndArray();

                        writer.Int64(pair.feature_index_1);
                        writer.Int64(pair.feature_index_2);

                        writer.Int64(pair.match_index);

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

                    writer.Key("relative_pose");
                    writer.StartArray();
                    for (const auto &pose : edge.payload.relative_poses)
                    {
                        writer.StartObject();

                        writer.Key("score");
                        writer.Int(pose.score);

                        writer.Key("orientation");
                        writer.StartArray();
                        for (int i = 0; i < 4; i++)
                        {
                            writer.Double(pose.orientation.coeffs()(i));
                        }
                        writer.EndArray();

                        writer.Key("position");
                        writer.StartArray();
                        for (int i = 0; i < 3; i++)
                        {
                            writer.Double(pose.position(i));
                        }
                        writer.EndArray();

                        writer.EndObject();
                    }
                    writer.EndArray();
                }
                writer.EndObject();
            }
            writer.EndObject();
        }
        writer.EndObject();

        return buffer.GetString();
    }
};
std::string serialize(const MeasurementGraph &graph)
{
    return Serializer<MeasurementGraph>::to_json(graph);
}

std::string toVisualizedGeoJson(const MeasurementGraph &graph,
                                std::function<Eigen::Vector3d(const Eigen::Vector3d &)> toGlobalCoordinates)
{
    return Serializer<MeasurementGraph>::visualizeGeoJson(graph, toGlobalCoordinates);
}

std::string toVisualizedGeoJson(const MeasurementGraph &graph, const std::vector<size_t> &node_ids,
                                const std::vector<size_t> &edge_ids,
                                std::function<Eigen::Vector3d(const Eigen::Vector3d &)> toGlobalCoordinates)
{
    return Serializer<MeasurementGraph>::visualizeGeoJson(graph, node_ids, edge_ids, toGlobalCoordinates);
}
} // namespace opencalibration
