#include "graph_utils.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <set>
#include <algorithm>
#include <sstream>
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
        cerr << "[ERROR] Broj cvorova mora biti >= 2!\n";
        return false;
    }
    
    if (numEdges < 0) {
        cerr << "[ERROR] Broj grana mora biti >= 0!\n";
        return false;
    }
    
    // Maksimalan broj grana u DAG-u je n*(n-1)/2
    long long maxPossibleEdges = (long long)numNodes * (numNodes - 1) / 2;
    if (numEdges > maxPossibleEdges) {
        cerr << "[ERROR] Previse grana za DAG sa " << numNodes 
             << " cvorova! Maksimalno: " << maxPossibleEdges << endl;
        return false;
    }
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> wDist(minW, maxW);
    
    cout << "[INFO] Generisanje " << numEdges << " grana..." << endl;
    
    vector<Edge> edges;
    edges.reserve(numEdges);
    
    // Za velike grafove koristimo efikasniji pristup
    if ((long long)numEdges > maxPossibleEdges / 2) {
        // PRISTUP 1: Generiši sve moguće grane i random shuffle
        cout << "[INFO] Koristim pristup sa generisanjem svih mogucih grana..." << endl;
        
        vector<pair<int,int>> allPairs;
        allPairs.reserve(maxPossibleEdges);
        
        for (int u = 0; u < numNodes - 1; ++u) {
            for (int v = u + 1; v < numNodes; ++v) {
                allPairs.push_back({u, v});
            }
            if (u % 10000 == 0) {
                cout << "  Generisanje parova: " << u << "/" << numNodes << "\r" << flush;
            }
        }
        cout << "  Generisanje parova: " << numNodes << "/" << numNodes << " [GOTOVO]" << endl;
        
        // Shuffle i uzmi prvih numEdges
        cout << "[INFO] Shuffle parova..." << flush;
        shuffle(allPairs.begin(), allPairs.end(), gen);
        cout << " Gotovo!" << endl;
        
        cout << "[INFO] Kreiranje grana sa tezinama..." << flush;
        for (int i = 0; i < numEdges; ++i) {
            int w = wDist(gen);
            edges.push_back({allPairs[i].first, allPairs[i].second, w});
            
            if ((i + 1) % 10000000 == 0) {
                cout << "\n  Kreirano: " << (i + 1) << "/" << numEdges << flush;
            }
        }
        cout << " Gotovo!" << endl;
        
    } else {
        // PRISTUP 2: Random sampling za manji broj grana
        cout << "[INFO] Koristim pristup sa random samplingom..." << endl;
        
        set<pair<int, int>> used;
        uniform_int_distribution<> nodeDist(0, numNodes - 1);
        
        int lastReport = 0;
        while ((int)edges.size() < numEdges) {
            int u = nodeDist(gen);
            int v = nodeDist(gen);
            
            if (u >= v) continue; // DAG pravilo
            if (used.count({u, v})) continue;
            
            used.insert({u, v});
            int w = wDist(gen);
            edges.push_back({u, v, w});
            
            // Progress report svakih 10%
            int progress = (edges.size() * 100) / numEdges;
            if (progress >= lastReport + 10) {
                cout << "  Progress: " << progress << "% (" 
                     << edges.size() << "/" << numEdges << ")" << endl;
                lastReport = progress;
            }
        }
    }
    
    cout << "[INFO] Snimanje u fajl..." << flush;
    
    // Snimi edge list (.txt)
    ofstream out(filename, ios::binary); // Binary za bržu IO
    if (!out.is_open()) {
        cerr << "\n[ERROR] Ne mogu otvoriti fajl " << filename << " za pisanje!\n";
        return false;
    }
    
    out << numNodes << " " << edges.size() << "\n";
    
    // Buffer za brže pisanje
    stringstream buffer;
    int bufferSize = 0;
    const int BUFFER_LIMIT = 100000;
    
    for (const auto& e : edges) {
        buffer << e.source << " " << e.destination << " " << e.weight << "\n";
        bufferSize++;
        
        if (bufferSize >= BUFFER_LIMIT) {
            out << buffer.str();
            buffer.str("");
            buffer.clear();
            bufferSize = 0;
        }
    }
    
    // Flush ostatak
    if (bufferSize > 0) {
        out << buffer.str();
    }
    
    out.close();
    cout << " Gotovo!" << endl;
    
    cout << "[INFO] Graf uspjesno kreiran: " << filename << endl;
    cout << "       Cvorova: " << numNodes << ", Grana: " << edges.size() << endl;
    
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