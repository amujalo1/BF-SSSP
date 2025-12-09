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
    const long INF_CHECK = numeric_limits<long>::max() / 4; 

    for (size_t i = 0; i < d1.size(); ++i) {
        bool is_inf_d1 = d1[i] > INF_CHECK;
        bool is_inf_d2 = d2[i] > INF_CHECK;

        if (is_inf_d1 && is_inf_d2) {
            continue;
        }
        if (d1[i] != d2[i]) {
            return false;
        }
    }
    return true;
}

// Struktura za čuvanje rezultata testova
struct TestResult {
    string version;
    string technique;
    double time;
    double speedup;
    bool correct;
};

// Funkcija za testiranje svih verzija na jednom grafu
vector<TestResult> test_all_versions(Graph* g, GraphSoA* gSoA, const string& graph_name) {
    vector<TestResult> results;
    int last_node = g->num_nodes - 1;
    
    cout << "\n" << string(80, '=') << endl;
    cout << "              TESTIRANJE NA GRAFU: " << graph_name << endl;
    cout << "              Cvorova: " << g->num_nodes << ", Grana: " << g->num_edges << endl;
    cout << string(80, '=') << endl;

    // Lambda za printanje rezultata
    auto print_results = [&](const string& name, const vector<long>& distances, 
                            double elapsed, double baseline_elapsed, const vector<long>& baseline_distances) {
        cout << "[VRIJEME] " << fixed << setprecision(3) << elapsed << " sekundi\n";
        
        const long INF_CHECK = numeric_limits<long>::max() / 4;
        if (distances[last_node] > INF_CHECK)
            cout << "  [REZULTAT] Cvor " << last_node << " nije dostupan iz izvora (0).\n";
        else
            cout << "  [REZULTAT] Najkraci put od 0 do " << last_node << " = " << distances[last_node] << endl;

        if (baseline_elapsed > 0) {
            double speedup = baseline_elapsed / elapsed;
            cout << "  [UBRZANJE] " << fixed << setprecision(2) << speedup << "x " 
                 << (speedup > 1.0 ? "(BRZE)" : (speedup < 1.0 ? "(SPORIJE)" : "(JEDNAKO)")) << endl; 

            if (name != "Originalna") {
                bool correct = compare_distances(baseline_distances, distances);
                cout << "  [KOREKTNOST] " << (correct ? "V TACNO" : "X NETACNO") << endl;
                return make_pair(speedup, correct);
            }
        }
        cout << string(80, '-') << endl;
        return make_pair(1.0, true);
    };

    // TEST 1: ORIGINALNA VERZIJA
    cout << "\n[TEST 1] ORIGINALNA VERZIJA (baseline - AoS, long)\n";
    cout << string(80, '=') << endl;
    
    auto start1 = chrono::high_resolution_clock::now();
    vector<long> distances1 = runBellmanFordSSSP(g, 0);
    auto end1 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed1 = end1 - start1;
    
    print_results("Originalna", distances1, elapsed1.count(), 0.0, distances1);
    results.push_back({"V1", "Originalna (Baseline)", elapsed1.count(), 1.0, true});

    // TEST 2: CACHE OPTIMIZACIJA
    cout << "\n[TEST 2] CACHE OPTIMIZACIJA (AoS, long)\n";
    cout << "  > Optimizacije: Sortiranje grana, Blocking, Prefetching\n";
    cout << string(80, '=') << endl;
    
    auto start2 = chrono::high_resolution_clock::now();
    vector<long> distances2 = runBellmanFordSSSP_CACHE(g, 0);
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed2 = end2 - start2;
    
    auto [speedup2, correct2] = print_results("Cache", distances2, elapsed2.count(), elapsed1.count(), distances1);
    results.push_back({"V2", "Cache Optim. (Sort+Block+Pref)", elapsed2.count(), speedup2, correct2});

    // TEST 3: OpenMP PARALELIZACIJA
    cout << "\n[TEST 3] OpenMP PARALELIZACIJA (AoS, long)\n";
    cout << "  > Optimizacije: V2 + OpenMP Multi-threading sa Atomic operacijama\n";
    cout << string(80, '=') << endl;
    
    auto start3 = chrono::high_resolution_clock::now();
    vector<long> distances3 = runBellmanFordSSSP_OMP(g, 0);
    auto end3 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed3 = end3 - start3;
    
    auto [speedup3, correct3] = print_results("OpenMP", distances3, elapsed3.count(), elapsed1.count(), distances1);
    results.push_back({"V3", "OpenMP (Multi-threading)", elapsed3.count(), speedup3, correct3});

    // TEST 4: SIMD AVX2 + OpenMP
    cout << "\n[TEST 4] SIMD (AVX2, 8-way) + OpenMP (SoA, int)\n";
    cout << "  > Optimizacije: AVX2 SIMD (8-way), SoA layout, OpenMP paralelizacija\n";
    cout << string(80, '=') << endl;

    auto start4 = chrono::high_resolution_clock::now();
    vector<int> distances4_i = runBellmanFordSSSP_SIMD_OMP(gSoA, 0);
    auto end4 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed4 = end4 - start4;
    vector<long> distances4 = to_long_vec(distances4_i);

    auto [speedup4, correct4] = print_results("SIMD_AVX2_OMP", distances4, elapsed4.count(), elapsed1.count(), distances1);
    results.push_back({"V4", "SIMD (AVX2, 8-way) + OpenMP", elapsed4.count(), speedup4, correct4});

    // TEST 5: SIMD AVX-512 + OpenMP
    cout << "\n[TEST 5] SIMD (AVX-512, 16-way) + OpenMP (SoA, int)\n";
    cout << "  > Optimizacije: AVX-512 SIMD (16-way), SoA layout, OpenMP paralelizacija\n";
    cout << string(80, '=') << endl;

    auto start5 = chrono::high_resolution_clock::now();
    vector<int> distances5_i = runBellmanFordSSSP_SIMD_512_OMP(gSoA, 0);
    auto end5 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed5 = end5 - start5;
    vector<long> distances5 = to_long_vec(distances5_i);

    auto [speedup5, correct5] = print_results("SIMD_AVX512_OMP", distances5, elapsed5.count(), elapsed1.count(), distances1);
    results.push_back({"V5", "SIMD (AVX-512, 16-way) + OpenMP", elapsed5.count(), speedup5, correct5});

    return results;
}

// Funkcija za ispis rezimea jednog grafa
void print_summary(const string& graph_name, const vector<TestResult>& results) {
    cout << "\n" << string(80, '#') << endl;
    cout << "          REZIME ZA GRAF: " << graph_name << endl;
    cout << string(80, '#') << endl;
    
    cout << "\n| Verzija | Tehnika                         | Vrijeme (s) | Ubrzanje (x) | Korektnost |\n";
    cout << "|:--------|:--------------------------------|:------------|:-------------|:-----------|\n";
    
    for (const auto& r : results) {
        cout << "| " << left << setw(7) << r.version 
             << " | " << setw(31) << r.technique
             << " | " << right << setw(11) << fixed << setprecision(3) << r.time 
             << " | " << setw(12) << setprecision(2) << r.speedup 
             << " | " << (r.correct ? "V TACNO   " : "X NETACNO ") << " |\n";
    }
}

int main() {
    string folder = "graph";
    
    // Definicije tri grafa
    struct GraphConfig {
        string name;
        string filename;
        int nodes;
        int edges;
        int weight_min;
        int weight_max;
    };
    
    vector<GraphConfig> graph_configs = {
        {"SPARSE (Rijetki)", folder + "/graf_rare.txt", 200000, 800000, -15, 35},
        {"NORMAL (Defaultni)", folder + "/graf.txt", 200000, 8000000, -15, 35},
        {"DENSE (Gusti)", folder + "/graf_dense.txt", 200000, 80000000, -15, 35}
    };
    
    // Kreiranje foldera
    if (!fs::exists(folder)) {
        fs::create_directory(folder);
        cout << "[INFO] Kreiran folder: " << folder << endl;
    }
    
    // Generisanje grafova ako ne postoje
    for (const auto& config : graph_configs) {
        if (!fs::exists(config.filename)) {
            cout << "[INFO] Generisem " << config.name << " graf (" 
                 << config.nodes << " cvorova, " << config.edges << " grana)..." << endl;
            bool ok = createGraph(config.filename, config.nodes, config.edges, 
                                config.weight_min, config.weight_max);
            if (!ok) {
                cerr << "[ERROR] Neuspjesno generisanje grafa: " << config.name << endl;
                return 1;
            }
        } else {
            cout << "[INFO] Graf " << config.name << " vec postoji --> preskacem generisanje." << endl;
        }
    }
    
    // Testiranje svih verzija na svim grafovima
    vector<vector<TestResult>> all_results;
    
    for (const auto& config : graph_configs) {
        cout << "\n\n" << string(80, '*') << endl;
        cout << "       UCITAVANJE I TESTIRANJE: " << config.name << endl;
        cout << string(80, '*') << endl;
        
        // Učitavanje grafa
        Graph* g = readGraph(config.filename);
        if (!g) {
            cerr << "[ERROR] Ne mogu ucitati graf (AoS): " << config.name << endl;
            continue;
        }
        
        GraphSoA* gSoA = readGraphSoA(config.filename);
        if (!gSoA) {
            cerr << "[ERROR] Ne mogu ucitati graf (SoA): " << config.name << endl;
            delete[] g->edge;
            delete g;
            continue;
        }
        
        // Testiranje svih verzija
        vector<TestResult> results = test_all_versions(g, gSoA, config.name);
        all_results.push_back(results);
        
        // Ispis rezimea za ovaj graf
        print_summary(config.name, results);
        
        // Čišćenje memorije
        delete[] g->edge;
        delete g;
        delete gSoA;
    }
    
    // FINALNI REZIME - Uporedni prikaz svih grafova
    cout << "\n\n" << string(100, '#') << endl;
    cout << "                         FINALNI UPOREDNI REZIME                         \n";
    cout << string(100, '#') << endl;
    
    for (size_t i = 0; i < graph_configs.size(); ++i) {
        cout << "\n=== " << graph_configs[i].name << " ===" << endl;
        cout << "Najbolje ubrzanje: ";
        double max_speedup = 1.0;
        string best_version;
        for (const auto& r : all_results[i]) {
            if (r.speedup > max_speedup) {
                max_speedup = r.speedup;
                best_version = r.version + " (" + r.technique + ")";
            }
        }
        cout << fixed << setprecision(2) << max_speedup << "x --> " << best_version << endl;
    }
    
    cout << "\n[INFO] Svi testovi zavrseni!" << endl;
    
    return 0;
}