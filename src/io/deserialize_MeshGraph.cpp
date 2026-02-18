#include <opencalibration/io/deserialize.hpp>

namespace
{
template <typename Out> void split(const std::string &s, char delim, Out result)
{
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim))
    {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}
} // namespace

namespace opencalibration
{
template <> class Deserializer<MeshGraph>
{
  public:
    static bool from_ply(std::istream &ply, MeshGraph &graph)
    {

        graph._edge_id_from_nodes_lookup.clear();
        graph._edges.clear();
        graph._nodes.clear();

        std::string line;

#define EXPECT_LINE(x)                                                                                                 \
    if (!std::getline(ply, line))                                                                                      \
        return false;                                                                                                  \
    if (line != x)                                                                                                     \
    return false

        EXPECT_LINE("ply");
        EXPECT_LINE("format ascii 1.0");
        EXPECT_LINE("comment exported from OpenCalibration");

        // read line with number of nodes
        std::getline(ply, line);
        if (line.substr(0, 15) != "element vertex ")
            return false;
        size_t num_nodes = std::stoll(line.substr(15));
        graph._nodes.reserve(num_nodes);

        EXPECT_LINE("property double x");
        EXPECT_LINE("property double y");
        EXPECT_LINE("property double z");
        EXPECT_LINE("property int nodeIndex");

        // read line with number of faces
        std::getline(ply, line);
        if (line.substr(0, 13) != "element face ")
            return false;
        size_t num_faces = std::stoll(line.substr(13));
        (void)num_faces;

        EXPECT_LINE("property list uchar int vertex_index");

        // read line with number of edges
        std::getline(ply, line);
        if (line.substr(0, 13) != "element edge ")
            return false;
        size_t num_edges = std::stoll(line.substr(13));
        graph._edges.reserve(num_edges);
        graph._edge_id_from_nodes_lookup.reserve(num_edges);

        EXPECT_LINE("property int vertex1");
        EXPECT_LINE("property int vertex2");
        EXPECT_LINE("property int edgeIndex");
        EXPECT_LINE("property uchar border");
        EXPECT_LINE("property int oppositeCorner1");
        EXPECT_LINE("property int oppositeCorner2");
        EXPECT_LINE("end_header");

        std::vector<size_t> sequential_node_ids;
        sequential_node_ids.reserve(num_nodes);
        for (size_t i = 0; i < num_nodes; i++)
        {
            if (!std::getline(ply, line))
                return false;
            auto words = split(line, ' ');
            if (words.size() != 4)
                return false;
            Eigen::Vector3d loc;
            for (int j = 0; j < 3; j++)
                loc[j] = std::stod(words[j]);
            size_t node_id = std::stoull(words[3]);
            MeshNode payload{loc};
            graph._nodes.emplace(node_id, std::move(payload));
            sequential_node_ids.push_back(node_id);
        }

        for (size_t i = 0; i < num_faces; i++)
        {
            if (!std::getline(ply, line))
                return false;
            auto words = split(line, ' ');
            if (words.size() != 4)
                return false;
        }

        for (size_t i = 0; i < num_edges; i++)
        {
            if (!std::getline(ply, line))
                return false;
            auto words = split(line, ' ');
            if (words.size() != 6)
                return false;
            size_t sequential_node = std::stoull(words[0]);
            if (sequential_node >= sequential_node_ids.size())
                return false;
            size_t source_id = sequential_node_ids[sequential_node];
            sequential_node = std::stoull(words[1]);
            if (sequential_node >= sequential_node_ids.size())
                return false;
            size_t dest_id = sequential_node_ids[sequential_node];
            size_t edge_id = std::stoull(words[2]);
            MeshEdge edge;
            edge.border = static_cast<bool>(std::stoi(words[3]));
            edge.triangleOppositeNodes = {std::stoull(words[4]), std::stoull(words[5])};

            graph._edges.emplace(edge_id, MeshGraph::Edge(edge, source_id, dest_id));
            graph._edge_id_from_nodes_lookup.emplace(MeshGraph::SourceDestIndex{source_id, dest_id}, edge_id);
            auto niter = graph._nodes.find(source_id);
            if (niter == graph._nodes.end())
                return false;
            niter->second._edges.insert(edge_id);
            niter = graph._nodes.find(dest_id);
            if (niter == graph._nodes.end())
                return false;
            niter->second._edges.insert(edge_id);
        }

#undef EXPECT_LINE

        if (std::getline(ply, line))
            return false;

        return true;
    }
};

bool deserialize(std::istream &ply, MeshGraph &graph)
{
    return Deserializer<MeshGraph>::from_ply(ply, graph);
}
} // namespace opencalibration
