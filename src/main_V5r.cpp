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
    string txtFile = folder + "/graf_dense.txt";

    // --- INICIJALIZACIJA (SoA) ---
    if (!fs::exists(folder)) { fs::create_directory(folder); }
    if (!fs::exists(txtFile)) { 
        createGraph(txtFile, 200000, 800000, -15, 35); 
    }
    GraphSoA* gSoA = readGraphSoA(txtFile);
    if (!gSoA) { cerr << "[ERROR] Ne mogu ucitati graf (SoA)!" << endl; cleanup(nullptr, gSoA); return 1; }

    cout << "[INFO] Testiram: V5 - SIMD Tiling (AVX2) + OpenMP\n";
    cout << "[INFO] Cvorovi: " << gSoA->num_nodes << ", Grane: " << gSoA->num_edges << endl;
    cout << string(60, '=') << endl;
    
    int last_node = gSoA->num_nodes - 1;

    // =======================================================
    // ========== TEST: V5 - SIMD AVX2 + OpenMP ==========
    // =======================================================
    auto start = chrono::high_resolution_clock::now();
    vector<int> distances_i = runBellmanFordSSSP_SIMD_512_OMP(gSoA, 0); // <-- KLJUČNA LINIJA
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    long result = (long)distances_i[last_node];
    cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed.count() << " sekundi\n";
    if (result >= numeric_limits<int>::max() / 4)
        cout << "[REZULTAT] Cvor " << last_node << " nije dostupan.\n";
    else
        cout << "[REZULTAT] Najkraci put (0 -> " << last_node << ") = " << result << endl;
    
    cleanup(nullptr, gSoA);
    return 0;
}