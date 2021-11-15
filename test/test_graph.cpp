#include <opencalibration/types/graph.hpp>

#include <gtest/gtest.h>

using namespace opencalibration;

TEST(graph, compiles)
{
    DirectedGraph<int, int> g;
}

TEST(graph, can_reset)
{
    DirectedGraph<int, int> g;
    g.addNode(42);

    EXPECT_EQ(g.size_nodes(), 1);

    g = DirectedGraph<int, int>();

    EXPECT_EQ(g.size_nodes(), 0);
}

TEST(graph, add_node)
{
    // GIVEN: a graph
    DirectedGraph<int, int> g;

    // WHEN: we add a node
    size_t node_id = g.addNode(42);

    // THEN: the node should be there
    const auto *node_val = g.getNode(node_id);
    ASSERT_NE(node_val, nullptr);
    EXPECT_EQ(node_val->payload, 42);
}

TEST(graph, add_nodes_and_edge)
{
    // GIVEN: a graph with 2 nodes
    DirectedGraph<int, int> g;
    size_t node_id = g.addNode(42);
    size_t node_id2 = g.addNode(108);

    // WHEN: we add an edge connecting them
    size_t edge_id = g.addEdge(int(9001), node_id, node_id2);

    // THEN: the edge should be there, and listed in the nodes
    const auto *edge_val = g.getEdge(edge_id);
    ASSERT_NE(edge_val, nullptr);
    EXPECT_EQ(edge_val->payload, 9001);
    EXPECT_EQ(edge_val->getSource(), node_id);
    EXPECT_EQ(edge_val->getDest(), node_id2);
    EXPECT_EQ(std::count(g.getNode(node_id)->getEdges().begin(), g.getNode(node_id)->getEdges().end(), edge_id), 1);
    EXPECT_EQ(std::count(g.getNode(node_id2)->getEdges().begin(), g.getNode(node_id2)->getEdges().end(), edge_id), 1);

    auto g2 = g;
    EXPECT_EQ(g2, g);

    // THEN: the edge at the two node ids should be the same too
    const auto *edge2 = g.getEdge(node_id, node_id2);
    EXPECT_EQ(edge_val, edge2);

    const auto &g3 = g;
    const auto *edge3 = g3.getEdge(node_id, node_id2);
    EXPECT_EQ(edge_val, edge3);
}
