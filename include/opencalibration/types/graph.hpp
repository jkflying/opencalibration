#pragma once

#include <ankerl/unordered_dense.h>
#include <deque>
#include <functional>
#include <optional>
#include <random>

namespace opencalibration
{
template <typename T> class Serializer;
template <typename T> class Deserializer;

template <typename NodePayload, typename EdgePayload> class DirectedGraph
{
    struct SourceDestIndex;

  public:
    class Node
    {
      public:
        Node(NodePayload &&p) : payload(std::move(p))
        {
        }

        NodePayload payload;

        [[nodiscard]] const ankerl::unordered_dense::set<size_t> &getEdges() const
        {
            return _edges;
        }

        bool operator==(const Node &other) const
        {
            return _edges == other._edges && payload == other.payload;
        }

      private:
        ankerl::unordered_dense::set<size_t> _edges;
        friend DirectedGraph;

        friend Serializer<DirectedGraph>;
        friend Deserializer<DirectedGraph>;
    };

    class Edge
    {
      public:
        Edge(EdgePayload p, size_t source, size_t dest) : payload(std::move(p)), _source(source), _dest(dest)
        {
        }

        EdgePayload payload;

        [[nodiscard]] size_t getSource() const
        {
            return _source;
        }
        [[nodiscard]] size_t getDest() const
        {
            return _dest;
        }

        bool operator==(const Edge &other) const
        {
            return _source == other._source && _dest == other._dest && payload == other.payload;
        }

      private:
        size_t _source;
        size_t _dest;
    };

    size_t addNode(NodePayload node_payload)
    {
        size_t identifier = distribution(generator);
        while (_nodes.count(identifier) > 0)
        {
            identifier = distribution(generator);
        }
        _nodes.emplace(identifier, std::move(node_payload));

        return identifier;
    }

    size_t addEdge(EdgePayload edge_payload, size_t source, size_t dest)
    {
        size_t identifier = distribution(generator);
        Edge e(std::move(edge_payload), source, dest);
        while (_edges.emplace(identifier, e).second == false)
        {
            identifier = distribution(generator);
        }

        _nodes.find(source)->second._edges.insert(identifier);
        _nodes.find(dest)->second._edges.insert(identifier);
        _edge_id_from_nodes_lookup.emplace(SourceDestIndex{source, dest}, identifier);

        return identifier;
    }

    [[nodiscard]] const Node *getNode(size_t node_id) const
    {
        auto iter = _nodes.find(node_id);
        if (iter == _nodes.end())
        {
            return nullptr;
        }
        return &(iter->second);
    }

    Node *getNode(size_t node_id)
    {
        auto iter = _nodes.find(node_id);
        if (iter == _nodes.end())
        {
            return nullptr;
        }
        return &(iter->second);
    }

    [[nodiscard]] const Edge *getEdge(size_t edge_id) const
    {
        auto iter = _edges.find(edge_id);
        if (iter == _edges.end())
        {
            return nullptr;
        }
        return &(iter->second);
    }

    Edge *getEdge(size_t edge_id)
    {
        auto iter = _edges.find(edge_id);
        if (iter == _edges.end())
        {
            return nullptr;
        }
        return &(iter->second);
    }

    [[nodiscard]] const Edge *getEdge(size_t source_node_id, size_t dest_node_id) const
    {
        SourceDestIndex idx{source_node_id, dest_node_id};
        auto iter = _edge_id_from_nodes_lookup.find(idx);
        if (iter != _edge_id_from_nodes_lookup.end())
        {
            return getEdge(iter->second);
        }
        return nullptr;
    }

    Edge *getEdge(size_t source_node_id, size_t dest_node_id)
    {
        SourceDestIndex idx{source_node_id, dest_node_id};
        auto iter = _edge_id_from_nodes_lookup.find(idx);
        if (iter != _edge_id_from_nodes_lookup.end())
        {
            return getEdge(iter->second);
        }
        return nullptr;
    }

    bool removeNode(size_t identifier)
    {
        auto nodeIter = _nodes.find(identifier);
        if (nodeIter == _nodes.end())
        {
            return false;
        }

        // Remove all edges connected to this node
        std::vector<size_t> edgesToRemove(nodeIter->second._edges.begin(), nodeIter->second._edges.end());
        for (size_t edgeId : edgesToRemove)
        {
            removeEdge(edgeId);
        }

        _nodes.erase(nodeIter);
        return true;
    }

    bool removeEdge(size_t identifier)
    {
        auto edgeIter = _edges.find(identifier);
        if (edgeIter == _edges.end())
        {
            return false;
        }

        size_t source = edgeIter->second.getSource();
        size_t dest = edgeIter->second.getDest();

        // Remove edge ID from source and destination nodes
        auto srcNodeIter = _nodes.find(source);
        if (srcNodeIter != _nodes.end())
        {
            srcNodeIter->second._edges.erase(identifier);
        }

        auto dstNodeIter = _nodes.find(dest);
        if (dstNodeIter != _nodes.end())
        {
            dstNodeIter->second._edges.erase(identifier);
        }

        // Remove from lookup
        _edge_id_from_nodes_lookup.erase(SourceDestIndex{source, dest});

        // Remove the edge
        _edges.erase(edgeIter);
        return true;
    }

    using NodeIterator = typename ankerl::unordered_dense::map<size_t, Node>::iterator;
    NodeIterator nodebegin()
    {
        return _nodes.begin();
    }
    NodeIterator nodeend()
    {
        return _nodes.end();
    }

    using EdgeIterator = typename ankerl::unordered_dense::map<size_t, Edge>::iterator;
    EdgeIterator edgebegin()
    {
        return _edges.begin();
    }
    EdgeIterator edgeend()
    {
        return _edges.end();
    }

    using CNodeIterator = typename ankerl::unordered_dense::map<size_t, Node>::const_iterator;
    [[nodiscard]] CNodeIterator cnodebegin() const
    {
        return _nodes.cbegin();
    }
    [[nodiscard]] CNodeIterator cnodeend() const
    {
        return _nodes.cend();
    }

    using CEdgeIterator = typename ankerl::unordered_dense::map<size_t, Edge>::const_iterator;
    [[nodiscard]] CEdgeIterator cedgebegin() const
    {
        return _edges.cbegin();
    }
    [[nodiscard]] CEdgeIterator cedgeend() const
    {
        return _edges.cend();
    }

    bool operator==(const DirectedGraph &other) const
    {
        return _nodes == other._nodes && _edges == other._edges &&
               _edge_id_from_nodes_lookup == other._edge_id_from_nodes_lookup;
    }

    [[nodiscard]] size_t size_nodes() const
    {
        return _nodes.size();
    }
    [[nodiscard]] size_t size_edges() const
    {
        return _edges.size();
    }

  private:
    struct SourceDestIndex
    {
        size_t source_id, dest_id;

        bool operator==(const SourceDestIndex &other) const
        {
            return source_id == other.source_id && dest_id == other.dest_id;
        }

        // hash operator
        size_t operator()(const SourceDestIndex &x) const
        {
            return x.source_id ^ x.dest_id;
        }
    };

    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution;
    ankerl::unordered_dense::map<size_t, Node> _nodes;
    ankerl::unordered_dense::map<size_t, Edge> _edges;
    ankerl::unordered_dense::map<SourceDestIndex, size_t, SourceDestIndex> _edge_id_from_nodes_lookup;

    friend Serializer<DirectedGraph>;
    friend Deserializer<DirectedGraph>;
};
} // namespace opencalibration
