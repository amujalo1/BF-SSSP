#include "graph_utils.h"
#include "bellman_ford.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <limits>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

void cleanup(Graph* g, GraphSoA* gSoA) {
    if (g) { delete[] g->edge; delete g; }
    if (gSoA) { delete gSoA; }
}

int main() {
    string folder = "graph";
    string txtFile = folder + "/graf_rare.txt";

    // --- INICIJALIZACIJA (AoS) ---
    if (!fs::exists(folder)) { fs::create_directory(folder); }
    if (!fs::exists(txtFile)) { 
        createGraph(txtFile, 200000, 8000000, -15, 35); 
    }

    Graph* g = readGraph(txtFile);
    if (!g) { cerr << "[ERROR] Ne mogu ucitati graf (AoS)!" << endl; cleanup(g, nullptr); return 1; }

    cout << "[INFO] Testiram: V3 - OpenMP Paralelizacija\n";
    cout << "[INFO] Cvorovi: " << g->num_nodes << ", Grane: " << g->num_edges << endl;
    cout << string(60, '=') << endl;
    
    int last_node = g->num_nodes - 1;

    // ===========================================
    // ========== TEST: V3 - OpenMP ==========
    // ===========================================
    auto start = chrono::high_resolution_clock::now();
    vector<long> distances = runBellmanFordSSSP_OMP(g, 0); // <-- KLJUČNA LINIJA
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed.count() << " sekundi\n";
    if (distances[last_node] >= numeric_limits<long>::max() / 4)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan.\n";
    else
        cout << "[REZULTAT] Najkraci put (0 -> " << last_node << ") = " << distances[last_node] << endl;
    
    cleanup(g, nullptr);
    return 0;
}