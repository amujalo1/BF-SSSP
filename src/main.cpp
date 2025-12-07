#include "graph_utils.h"
#include "bellman_ford.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

// Pomoćna funkcija za poređenje rezultata
static vector<long> to_long_vec(const vector<int>& v) {
    vector<long> out; out.reserve(v.size());
    for (int x : v) out.push_back((long)x); 
    return out;
}

// Pomoćna funkcija za provjeru korektnosti s obzirom na INF vrijednost
bool compare_distances(const vector<long>& d1, const vector<long>& d2) {
    if (d1.size() != d2.size()) return false;
    // Opreznija provjera za Long/Int MAX vrijednosti
    const long INF_CHECK = numeric_limits<long>::max() / 4; 

    for (size_t i = 0; i < d1.size(); ++i) {
        // Ako su obje "beskonačne" ili blizu, smatramo ih jednakim
        bool is_inf_d1 = d1[i] > INF_CHECK;
        bool is_inf_d2 = d2[i] > INF_CHECK;

        if (is_inf_d1 && is_inf_d2) {
            continue;
        }
        // Vrijednosti se moraju tačno poklapati
        if (d1[i] != d2[i]) {
            return false;
        }
    }
    return true;
}


int main() {
    string folder = "graph";
    string txtFile = folder + "/graf.txt";
    
    // Kreiranje foldera
    if (!fs::exists(folder)) {
        fs::create_directory(folder);
        cout << "[INFO] Kreiran folder: " << folder << endl;
    }
    
    // Kreiranje grafa
    if (!fs::exists(txtFile)) {
        cout << "[INFO] Fajl ne postoji, generisem graf (200K cvorova, 8M grana, tezine [-15, 35])..." << endl;
        bool ok = createGraph(txtFile, 200000, 8000000, -15, 35); 
        if (!ok) {
            cerr << "[ERROR] Neuspjesno generisanje grafa!" << endl;
            return 1;
        }
    } else {
        cout << "[INFO] Graf vec postoji --> preskacem generisanje." << endl;
    }
    
    // Učitavanje grafa za AoS i SoA formate
    Graph* g = readGraph(txtFile); // AoS (za V1-V3)
    if (!g) {
        cerr << "[ERROR] Ne mogu ucitati graf (AoS)!" << endl;
        return 1;
    }
    
    GraphSoA* gSoA = readGraphSoA(txtFile); // SoA (za V4-V7)
    if (!gSoA) {
        cerr << "[ERROR] Ne mogu ucitati graf (SoA)!" << endl;
        delete[] g->edge;
        delete g;
        return 1;
    }
    
    cout << "[INFO] Ucitano: " << g->num_nodes << " cvorova, "
         << g->num_edges << " grana.\n";
    cout << string(80, '=') << endl;
    
    int last_node = g->num_nodes - 1;
    vector<long> distances1; // Koristit cemo V1 kao referentnu tacku
    
    // Lambda za printanje rezultata
    auto print_results = [&](const string& name, const vector<long>& distances, 
                            double elapsed, double baseline_elapsed) {
        cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed << " sekundi\n";
        
        // Prikaz putanje do zadnjeg cvora
        const long INF_CHECK = numeric_limits<long>::max() / 4;
        if (distances[last_node] > INF_CHECK)
            cout << "  [REZULTAT] Cvor " << last_node << " nije dostupan iz izvora (0).\n";
        else
            cout << "  [REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances[last_node] << endl;

        // Ubrzanje i korektnost
        if (baseline_elapsed > 0) {
            double speedup = baseline_elapsed / elapsed;
            cout << "  [UBRZANJE] " << fixed << setprecision(2) << speedup << "x " 
                 << (speedup > 1.0 ? "(BRZE)" : "(SPORIJE)") << endl; 

            if (name != "Originalna") {
                cout << "  [KOREKTNOST] " << (compare_distances(distances1, distances) ? "V TACNO" : "X NETACNO") << endl; 
            }
        }
        cout << string(80, '-') << endl;
    };


    // =================================================================================
    // ========== TEST 1: ORIGINALNA VERZIJA (Baseline) ================================
    // =================================================================================
    cout << "\n[TEST 1] ORIGINALNA VERZIJA (baseline - AoS, long)\n";
    cout << string(80, '=') << endl;
    
    auto start1 = chrono::high_resolution_clock::now();
    distances1 = runBellmanFordSSSP(g, 0);
    auto end1 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed1 = end1 - start1;
    
    print_results("Originalna", distances1, elapsed1.count(), 0.0);

    // =================================================================================
    // ========== TEST 2: CACHE OPTIMIZACIJA ===========================================
    // =================================================================================
    cout << "\n[TEST 2] CACHE OPTIMIZACIJA (AoS, long)\n";
    cout << "  > Optimizacije: Sortiranje grana, Blocking, Prefetching\n";
    cout << string(80, '=') << endl;
    
    auto start2 = chrono::high_resolution_clock::now();
    vector<long> distances2 = runBellmanFordSSSP_CACHE(g, 0);
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed2 = end2 - start2;
    
    print_results("Cache", distances2, elapsed2.count(), elapsed1.count());

    // =================================================================================
    // ========== TEST 3: OpenMP PARALELIZACIJA ========================================
    // =================================================================================
    cout << "\n[TEST 3] OpenMP PARALELIZACIJA (AoS, long)\n";
    cout << "  > Optimizacije: V2 + OpenMP Multi-threading sa Atomic operacijama\n";
    cout << string(80, '=') << endl;
    
    auto start3 = chrono::high_resolution_clock::now();
    vector<long> distances3 = runBellmanFordSSSP_OMP(g, 0);
    auto end3 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed3 = end3 - start3;
    
    print_results("OpenMP", distances3, elapsed3.count(), elapsed1.count());

    // =================================================================================
    // ========== TEST 4: SIMD TILING AVX2 (Prva SIMD) =================================
    // =================================================================================
    cout << "\n[TEST 4] SIMD TILING (AVX2 - SoA, int)\n";
    cout << "  > Optimizacije: Destination Tiling, SoA, AVX2 SIMD (8-way)\n";
    cout << string(80, '=') << endl;
    
    auto start4 = chrono::high_resolution_clock::now();
    vector<int> distances4_i = runBellmanFordSSSP_SIMD_Tiling(gSoA, 0);
    auto end4 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed4 = end4 - start4;
    vector<long> distances4 = to_long_vec(distances4_i);
    
    print_results("SIMD_Tiling_AVX2", distances4, elapsed4.count(), elapsed1.count());

    // =================================================================================
    // ========== TEST 5: SIMD TILING AVX2 + OpenMP ====================================
    // =================================================================================
    cout << "\n[TEST 5] SIMD TILING + OpenMP (AVX2 - SoA, int)\n";
    cout << "  > Optimizacije: V4 + OpenMP paralelizacija sa Atomic operacijama\n";
    cout << string(80, '=') << endl;
    
    auto start5 = chrono::high_resolution_clock::now();
    vector<int> distances5_i = runBellmanFordSSSP_SIMD_Tiling_OMP(gSoA, 0);
    auto end5 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed5 = end5 - start5;
    vector<long> distances5 = to_long_vec(distances5_i);
    
    print_results("SIMD_Tiling_AVX2_OMP", distances5, elapsed5.count(), elapsed1.count());



    // ==================== FINALNI REZIME ====================
    cout << "\n" << string(80, '#') << endl;
    cout << "                  REZIME POSTEPENIH OPTIMIZACIJA                  \n";
    cout << string(80, '#') << endl;
    cout << fixed << setprecision(3);
    
    cout << "\n| Verzija | Tehnika | Vrijeme (s) | Ubrzanje (x) | Korektnost |\n";
    cout << "|:--------|:--------------------------------|:------------|:-------------|:-----------|\n";
    cout << "| V1      | Originalna (Baseline)           | " << setw(11) << elapsed1.count() << " | N/A          | N/A        |\n";
    cout << "| V2      | Cache Optim. (Sort+Block+Pref)  | " << setw(11) << elapsed2.count() << " | " << setw(12) << setprecision(2) << (elapsed1.count()/elapsed2.count()) << " | " << (compare_distances(distances1, distances2) ? "V TACNO" : "X NETACNO") << " |\n";
    cout << "| V3      | OpenMP (Multi-threading)        | " << setw(11) << elapsed3.count() << " | " << setw(12) << setprecision(2) << (elapsed1.count()/elapsed3.count()) << " | " << (compare_distances(distances1, distances3) ? "V TACNO" : "X NETACNO") << " |\n";
    cout << "| V4      | SIMD Tiling (AVX2, 8-way)       | " << setw(11) << elapsed4.count() << " | " << setw(12) << setprecision(2) << (elapsed1.count()/elapsed4.count()) << " | " << (compare_distances(distances1, distances4) ? "V TACNO" : "X NETACNO") << " |\n";
    cout << "| V5      | V4 + OpenMP                     | " << setw(11) << elapsed5.count() << " | " << setw(12) << setprecision(2) << (elapsed1.count()/elapsed5.count()) << " | " << (compare_distances(distances1, distances5) ? "V TACNO" : "X NETACNO") << " |\n";
    
    
    // Čišćenje memorije
    delete[] g->edge;
    delete g;
    delete gSoA;
    
    return 0;
}
