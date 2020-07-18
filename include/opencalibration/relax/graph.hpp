#pragma once

#include <functional>
#include <optional>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace opencalibration
{

template <typename NodePayload, typename EdgePayload> class DirectedGraph
{
  public:
    class Node
    {
      public:
        Node(NodePayload &&p) : payload(std::move(p))
        {
        }

        NodePayload payload;

        const std::unordered_set<size_t> &getEdges() const
        {
            return _edges;
        }

      private:
        std::unordered_set<size_t> _edges;
        friend DirectedGraph;
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
        Edge e (std::move(edge_payload), source, dest);
        while (_edges.emplace(identifier, std::move(e)).second == false)
        {
            identifier = distribution(generator);
        }

        _nodes.find(source)->second._edges.insert(identifier);
        _nodes.find(dest)->second._edges.insert(identifier);

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

    bool removeNode(size_t identifier);
    bool removeEdge(size_t identifier);

  private:
    std::default_random_engine generator;
    std::uniform_int_distribution<size_t> distribution;
    std::unordered_map<size_t, Node> _nodes;
    std::unordered_map<size_t, Edge> _edges;
};
} // namespace opencalibration
