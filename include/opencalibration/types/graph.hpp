#pragma once

#include <deque>
#include <functional>
#include <optional>
#include <random>
#include <unordered_map>

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

        const std::deque<size_t> &getEdges() const
        {
            return _edges;
        }

        bool operator==(const Node &other) const
        {
            return _edges == other._edges && payload == other.payload;
        }

      private:
        std::deque<size_t> _edges;
        friend DirectedGraph;

        friend Serializer<DirectedGraph>;
        friend Deserializer<DirectedGraph>;
    };

    class Edge
    {
      public:
        Edge(EdgePayload &&p, size_t source, size_t dest) : payload(std::move(p)), _source(source), _dest(dest)
        {
        }

        EdgePayload payload;

        size_t getSource() const
        {
            return _source;
        }
        size_t getDest() const
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

    size_t addNode(NodePayload &&node_payload)
    {
        size_t identifier = distribution(generator);
        while (_nodes.emplace(identifier, std::move(node_payload)).second == false)
        {
            identifier = distribution(generator);
        }

        return identifier;
    }

    size_t addEdge(EdgePayload &&edge_payload, size_t source, size_t dest)
    {
        size_t identifier = distribution(generator);
        Edge e(std::move(edge_payload), source, dest);
        while (_edges.emplace(identifier, e).second == false)
        {
            identifier = distribution(generator);
        }

        _nodes.find(source)->second._edges.push_back(identifier);
        _nodes.find(dest)->second._edges.push_back(identifier);
        _edge_id_from_nodes_lookup.emplace(SourceDestIndex{source, dest}, identifier);

        return identifier;
    }

    const Node *getNode(size_t node_id) const
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

    const Edge *getEdge(size_t edge_id) const
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

    const Edge *getEdge(size_t source_node_id, size_t dest_node_id) const
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

    bool removeNode(size_t identifier);
    bool removeEdge(size_t identifier);

    using NodeIterator = typename std::unordered_map<size_t, Node>::iterator;
    NodeIterator nodebegin()
    {
        return _nodes.begin();
    }
    NodeIterator nodeend()
    {
        return _nodes.end();
    }

    using EdgeIterator = typename std::unordered_map<size_t, Edge>::iterator;
    EdgeIterator edgebegin()
    {
        return _edges.begin();
    }
    EdgeIterator edgeend()
    {
        return _edges.end();
    }

    using CNodeIterator = typename std::unordered_map<size_t, Node>::const_iterator;
    CNodeIterator cnodebegin() const
    {
        return _nodes.cbegin();
    }
    CNodeIterator cnodeend() const
    {
        return _nodes.cend();
    }

    using CEdgeIterator = typename std::unordered_map<size_t, Edge>::const_iterator;
    CEdgeIterator cedgebegin() const
    {
        return _edges.cbegin();
    }
    CEdgeIterator cedgeend() const
    {
        return _edges.cend();
    }

    bool operator==(const DirectedGraph &other) const
    {
        return _nodes == other._nodes && _edges == other._edges &&
               _edge_id_from_nodes_lookup == other._edge_id_from_nodes_lookup;
    }

    size_t size_nodes() const
    {
        return _nodes.size();
    }
    size_t size_edges() const
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
    std::unordered_map<size_t, Node> _nodes;
    std::unordered_map<size_t, Edge> _edges;
    std::unordered_map<SourceDestIndex, size_t, SourceDestIndex> _edge_id_from_nodes_lookup;

    friend Serializer<DirectedGraph>;
    friend Deserializer<DirectedGraph>;
};
} // namespace opencalibration
