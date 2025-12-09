#ifndef BELLMAN_FORD_H
#define BELLMAN_FORD_H

#include <vector>
#include <limits>
#include <algorithm>
#include <atomic>
#include <tuple>
// Pretpostavlja se da graph_utils.h sadrži definicije za
// struct Graph i struct GraphSoA (koja ima std::vector<int> sources, destinations, weights)
#include "graph_utils.h" 
#include <immintrin.h> // Za AVX2 i AVX-512 instrukcije

// ============================================================
//           DEKLARACIJE BELLMAN-FORD VERZIJA
// ============================================================

// Verzija 1: Originalna (baseline) implementacija
// Koristi Aos (Array of Structs) format grana (Graph*) i long za distance.
std::vector<long> runBellmanFordSSSP(Graph* graph, int source_node_id);

// Verzija 2: Optimizacija za CACHE lokalnost
// Dodaje: sortiranje grana + blocking + prefetching. Koristi long za distance.
std::vector<long> runBellmanFordSSSP_CACHE(Graph* graph, int source_node_id);

// Verzija 3: OpenMP PARALELIZACIJA (CPU Multi-threading)
// Koristi OpenMP i std::atomic<long> za thread-safe relaksaciju. Koristi long za distance.
std::vector<long> runBellmanFordSSSP_OMP(Graph* graph, int source_node_id);

// Verzija 4: SIMD Tiling sa AVX2 (Prva SIMD verzija)
// Koristi SoA format (GraphSoA*) i int za distance (zbog SIMD širine registra i gather).
std::vector<int> runBellmanFordSSSP_SIMD_OMP(GraphSoA* graph, int source_node_id);

// Verzija 5: SIMD Tiling + OpenMP (AVX2)
// Kombinuje paralelizaciju (V3) i AVX2 SIMD (V4). Koristi int za distance.
std::vector<int> runBellmanFordSSSP_SIMD_512_OMP(GraphSoA* graph, int source_node_id);

std::vector<long> runBellmanFordSSSP_CACHE_SORT(Graph* graph, int source_node_id);

#endif // BELLMAN_FORD_H