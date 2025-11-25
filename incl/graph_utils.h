#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <string>
#include <vector>

struct Edge {
    int source;
    int destination;
    int weight;
};

struct Graph {
    int num_nodes;
    int num_edges;
    Edge* edge;
};

// Nova struktura za SoA format
struct GraphSoA {
    int num_nodes;
    int num_edges;
    std::vector<int> sources;
    std::vector<int> destinations;
    std::vector<int> weights;
};

bool createGraph(const std::string& filename,
                 int numNodes,
                 int numEdges,
                 int minW,
                 int maxW);

Graph* readGraph(const std::string& filename);

// Nova funkcija za čitanje u SoA format
GraphSoA* readGraphSoA(const std::string& filename);

#endif // GRAPH_UTILS_H