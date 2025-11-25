#include "graph_utils.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <set>
#include <algorithm>

using namespace std;

/* =====================================================
   GENERISANJE DAG GRAFA (bez negativnih ciklusa)
   Format izlaza:
       num_nodes num_edges
       u v w
       u v w
   ===================================================== */
bool createGraph(const string& filename,
                 int numNodes,
                 int numEdges,
                 int minW,
                 int maxW)
{
    if (numNodes < 2) {
        cerr << "Broj čvorova mora biti >= 2!\n";
        return false;
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> nodeDist(0, numNodes - 1);
    uniform_int_distribution<> wDist(minW, maxW);

    set<pair<int, int>> used;
    vector<Edge> edges;
    edges.reserve(numEdges);

    // Generišemo DAG tako što zahtijevamo u < v
    while ((int)edges.size() < numEdges) {
        int u = nodeDist(gen);
        int v = nodeDist(gen);

        if (u >= v) continue;           // DAG pravilo
        if (used.count({u, v})) continue;

        used.insert({u, v});
        int w = wDist(gen);

        edges.push_back({u, v, w});
    }

    // --- Snimi edge list (.txt)
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Ne mogu otvoriti fajl " << filename << " za pisanje!\n";
        return false;
    }

    out << numNodes << " " << numEdges << "\n";
    for (auto& e : edges) {
        out << e.source << " "
            << e.destination << " "
            << e.weight << "\n";
    }
    out.close();

    // --- Generiši DOT fajl
    string dotFile = filename.substr(0, filename.find_last_of('.')) + ".dot";
    ofstream dot(dotFile);
    if (!dot.is_open()) {
        cerr << "Ne mogu otvoriti DOT fajl " << dotFile << " za pisanje!\n";
        return false;
    }

    dot << "digraph G {\n";
    dot << "    rankdir=LR;\n"; // lijevo->desno (neuron-network stil)
    dot << "    node [shape=circle, fontsize=12];\n\n";

    // Opcionalno: označi izvor i ponor
    dot << "    0 [shape=doublecircle, color=green, label=\"source (0)\"];\n";
    dot << "    " << (numNodes-1) << " [shape=doublecircle, color=red, label=\"sink (" << numNodes-1 << ")\"];\n\n";

    // Ispisi sve čvorove
    for (int i = 0; i < numNodes; i++)
        dot << "    " << i << ";\n";

    dot << "\n";

    // Ispisi sve grane
    for (auto& e : edges) {
        dot << "    " << e.source << " -> " << e.destination
            << " [label=\"" << e.weight << "\"];\n";
    }

    dot << "}\n";
    dot.close();

    cout << "[INFO] Graf kreiran: " << filename 
         << " i DOT fajl: " << dotFile << endl;

    return true;
}

/* =====================================================
   UČITAVANJE GRAFA U STRUCT Graph
   Format ulaza:
        num_nodes num_edges
        u v w
        u v w
   ===================================================== */
Graph* readGraph(const string& filename)
{
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Ne mogu otvoriti fajl " << filename << "!\n";
        return nullptr;
    }

    Graph* g = new Graph;
    in >> g->num_nodes >> g->num_edges;

    g->edge = new Edge[g->num_edges];

    for (int i = 0; i < g->num_edges; ++i) {
        in >> g->edge[i].source
           >> g->edge[i].destination
           >> g->edge[i].weight;
    }

    return g;
}

GraphSoA* readGraphSoA(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Greska: Ne mogu otvoriti fajl " << filename << std::endl;
        return nullptr;
    }

    auto* gSoA = new GraphSoA();
    int N, E;
    if (!(file >> N >> E)) {
        delete gSoA;
        return nullptr;
    }
    gSoA->num_nodes = N;
    gSoA->num_edges = E;

    // Jednostavno učitavanje bez sortiranja
    gSoA->sources.reserve(E);
    gSoA->destinations.reserve(E);
    gSoA->weights.reserve(E);
    
    for (int i = 0; i < E; ++i) {
        int u, v, weight;
        if (!(file >> u >> v >> weight)) {
            delete gSoA;
            return nullptr;
        }
        gSoA->sources.push_back(u);
        gSoA->destinations.push_back(v);
        gSoA->weights.push_back(weight);
    }

    return gSoA;
}