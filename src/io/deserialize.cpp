#include <opencalibration/io/deserialize.hpp>

#include "base64.h"

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
        d.Parse<rapidjson::kParseFullPrecisionFlag>(json.c_str());

        if (d.IsObject())
        {
            const auto &base = d.GetObject();
            if (base.HasMember("version") && base["version"].IsInt64() && base["version"].GetInt64() == 1)
            {
                const auto &nodes = base["nodes"].GetObject();
                for (auto niter = nodes.begin(); niter != nodes.end(); ++niter)
                {
                    size_t node_id = std::atoll(niter->name.GetString());
                    image img;
                    img.path = niter->value.GetObject()["path"].GetString();

                    const auto &model = niter->value.GetObject()["model"].GetObject();
                    img.model.pixels_cols = model["dimensions"].GetArray()[0].GetInt64();
                    img.model.pixels_rows = model["dimensions"].GetArray()[1].GetInt64();
                    img.model.focal_length_pixels = model["focal_length"].GetDouble();
                    img.model.principle_point[0] = model["principal"].GetArray()[0].GetDouble();
                    img.model.principle_point[1] = model["principal"].GetArray()[1].GetDouble();
                    img.model.projection_type = "planar" == std::string(model["projection"].GetString())
                                                    ? CameraModel::ProjectionType::PLANAR
                                                    : CameraModel::ProjectionType::UNKNOWN;

                    MeasurementGraph::Node node(std::move(img));

                    const auto &edge_ids = niter->value.GetObject()["edges"].GetArray();
                    for (const auto &edge_id : edge_ids)
                    {
                        node._edges.push_back(std::atoll(edge_id.GetString()));
                    }

                    // TODO: metadata

                    const auto &features = niter->value.GetObject()["features"].GetArray();
                    std::string descriptor;
                    for (auto fiter = features.begin(); fiter != features.end(); ++fiter)
                    {
                        feature_2d f;
                        const char *base64_descriptor = fiter->GetObject()["descriptor"].GetString();
                        descriptor.resize(Base64decode_len(base64_descriptor), '\0');
                        int actual_size = Base64decode(const_cast<char *>(descriptor.c_str()), base64_descriptor);
                        assert(descriptor.size() == actual_size);
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
                }
            }
        }
        return false;
    }
};
bool deserialize(const std::string &json, MeasurementGraph &graph)
{
    return Deserializer<MeasurementGraph>::from_json(json, graph);
}
} // namespace opencalibration
