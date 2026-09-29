#include <opencalibration/io/serialize.hpp>

#include <opencalibration/geometry/utils.hpp>

#include <iostream>
#include <limits>

#include <algorithm>
#include <ankerl/unordered_dense.h>

namespace
{
struct ArrayHash
{
    template <size_t N> size_t operator()(const std::array<size_t, N> &arr) const
    {
        size_t result = arr.size();
        for (auto &i : arr)
        {
            result ^= i + 0x9e3779b9 + (result << 6) + (result >> 2);
        }
        return result;
    }
};

} // namespace
namespace opencalibration
{

std::vector<std::array<size_t, 3>> meshFaces(const MeshGraph &graph)
{
    ankerl::unordered_dense::set<std::array<size_t, 3>, ArrayHash> faces;
    for (auto it = graph.cedgebegin(); it != graph.cedgeend(); ++it)
    {
        const auto &edge = it->second;
        auto addFace = [&](size_t oppositeCorner) {
            std::array<size_t, 3> face{edge.getSource(), edge.getDest(), oppositeCorner};
            std::sort(face.begin(), face.end());
            std::array<Eigen::Vector3d, 3> corners;
            for (size_t i = 0; i < 3; i++)
                corners[i] = graph.getNode(face[i])->payload.location;
            if (anticlockwise(corners))
                std::swap(face[0], face[1]);
            faces.insert(face);
        };

        addFace(edge.payload.triangleOppositeNodes[0]);
        if (!edge.payload.border)
            addFace(edge.payload.triangleOppositeNodes[1]);
    }

    std::vector<std::array<size_t, 3>> sorted(faces.begin(), faces.end());
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}

template <> class Serializer<MeshGraph>
{
  public:
    static bool to_ply(const MeshGraph &graph, std::ostream &out)
    {
        const auto *const newline = "\n";
        out << "ply" << newline;
        out << "format ascii 1.0" << newline;
        out << "comment exported from OpenCalibration" << newline;
        out << "element vertex " << graph.size_nodes() << newline;
        out << "property double x" << newline;
        out << "property double y" << newline;
        out << "property double z" << newline;
        out << "property int nodeIndex" << newline;

        const auto faces = meshFaces(graph);
        std::vector<size_t> sortedEdges;
        sortedEdges.reserve(graph.size_edges());
        std::transform(graph.cedgebegin(), graph.cedgeend(), std::back_inserter(sortedEdges),
                       [](const auto &iter) { return iter.first; });
        std::sort(sortedEdges.begin(), sortedEdges.end());

        out << "element face " << faces.size() << newline;
        out << "property list uchar int vertex_index" << newline;

        out << "element edge " << graph.size_edges() << newline;
        out << "property int vertex1" << newline;
        out << "property int vertex2" << newline;
        out << "property int edgeIndex" << newline;
        out << "property uchar border" << newline;
        out << "property int oppositeCorner1" << newline;
        out << "property int oppositeCorner2" << newline;
        out << "end_header" << newline;

        ankerl::unordered_dense::map<size_t, size_t> nodeSequenceLookup;
        std::vector<size_t> sortedNodes;
        sortedNodes.reserve(graph.size_nodes());
        std::transform(graph.cnodebegin(), graph.cnodeend(), std::back_inserter(sortedNodes),
                       [](const auto &iter) { return iter.first; });
        std::sort(sortedNodes.begin(), sortedNodes.end());
        const auto originalPrecision = out.precision(std::numeric_limits<double>::max_digits10);
        for (size_t node_id : sortedNodes)
        {
            const auto &node = *graph.getNode(node_id);
            nodeSequenceLookup[node_id] = nodeSequenceLookup.size();
            const auto &loc = node.payload.location;
            out << loc[0] << " " << loc[1] << " " << loc[2] << " " << node_id << newline;
        }
        out.precision(originalPrecision);

        for (const auto &face : faces)
        {
            std::array<size_t, 3> mappedFace;
            for (size_t i = 0; i < face.size(); i++)
            {
                mappedFace[i] = nodeSequenceLookup[face[i]];
            }
            out << "3 " << mappedFace[0] << " " << mappedFace[1] << " " << mappedFace[2] << newline;
        }

        for (size_t edge_id : sortedEdges)
        {
            const auto &edge = *graph.getEdge(edge_id);
            size_t source = edge.getSource();
            size_t dest = edge.getDest();

            const auto &payload = edge.payload;

            out << nodeSequenceLookup[source] << " " << nodeSequenceLookup[dest] << " " << edge_id << " "
                << payload.border << " " << payload.triangleOppositeNodes[0] << " " << payload.triangleOppositeNodes[1]
                << newline;
        }

        return true;
    }
};

bool serialize(const MeshGraph &graph, std::ostream &out)
{
    return Serializer<MeshGraph>::to_ply(graph, out);
}
} // namespace opencalibration
