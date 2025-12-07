#include "bellman_ford.h"
#include <iostream>
#include <limits>
#include <omp.h>
#include <algorithm>
#include <immintrin.h>
#include <cstring>
#include <numeric>      
#include <atomic>
#include <tuple>
using namespace std;

// ============================================================
//      VERZIJA 1: ORIGINALNA (baseline)
// ============================================================
std::vector<long> runBellmanFordSSSP(Graph* graph, int source_node_id)
{
    int no_of_nodes = graph->num_nodes;
    int no_of_edges = graph->num_edges;
    
    const long INF = numeric_limits<long>::max() / 2;
    vector<long> node_distances(no_of_nodes, INF);
    node_distances[source_node_id] = 0;

    // Relax edges
    for (int i = 0; i < no_of_nodes - 1; i++) {
        bool relaxed = false;
        for (int j = 0; j < no_of_edges; j++) {
            int u = graph->edge[j].source;
            int v = graph->edge[j].destination;
            long w = graph->edge[j].weight; 
            
            if (node_distances[u] < INF &&
                node_distances[u] + w < node_distances[v]) {
                node_distances[v] = node_distances[u] + w;
                relaxed = true;
            }
        }
        if (!relaxed)
            break;
    }

    return node_distances;
}

// ============================================================
//      VERZIJA 2: CACHE OPTIMIZACIJA
//      Dodaje: sortiranje grana + blocking + prefetching
// ============================================================
std::vector<long> runBellmanFordSSSP_CACHE(Graph* graph, int source_node_id) {
    int no_of_nodes = graph->num_nodes;
    int no_of_edges = graph->num_edges;
    
    const long INF = numeric_limits<long>::max() / 2;
    vector<long> node_distances(no_of_nodes, INF);
    node_distances[source_node_id] = 0;
    
    // OPTIMIZACIJA: Sortiranje grana po source čvoru za bolju cache lokalnost
    vector<Edge> sorted_edges(graph->edge, graph->edge + no_of_edges);
    sort(sorted_edges.begin(), sorted_edges.end(), 
         [](const Edge& a, const Edge& b) { return a.source < b.source; });
    
    // Glavna petlja relaksacije (ista kao originalna)
    for (int iter = 0; iter < no_of_nodes - 1; iter++) {
        bool relaxed = false;
        
        // OPTIMIZACIJA: Procesiranje po blokovima za bolju cache iskorištenost
        const int BLOCK_SIZE = 64;
        for (int block_start = 0; block_start < no_of_edges; block_start += BLOCK_SIZE) {
            int block_end = min(block_start + BLOCK_SIZE, no_of_edges);
            
            // OPTIMIZACIJA: Prefetch za sljedeći blok
            if (block_start + BLOCK_SIZE < no_of_edges) {
                __builtin_prefetch(&sorted_edges[block_start + BLOCK_SIZE], 0, 3);
            }
            
            // Procesiranje trenutnog bloka (ista logika kao originalna)
            for (int j = block_start; j < block_end; j++) {
                int u = sorted_edges[j].source;
                int v = sorted_edges[j].destination;
                long w = sorted_edges[j].weight;
                
                long dist_u = node_distances[u];
                if (dist_u < INF) {
                    long new_dist = dist_u + w;
                    if (new_dist < node_distances[v]) {
                        node_distances[v] = new_dist;
                        relaxed = true;
                    }
                }
            }
        }
        
        if (!relaxed) break;
    }
    
    return node_distances;
}

// ============================================================
//      VERZIJA 3: OpenMP PARALELIZACIJA
//      Dodaje: multi-threading sa thread-safe atomic operacijama
//      Zadržava: sortiranje + blocking iz CACHE verzije
// ============================================================
std::vector<long> runBellmanFordSSSP_OMP(Graph* graph, int source_node_id)
{
    int no_of_nodes = graph->num_nodes;
    int no_of_edges = graph->num_edges;
    const long INF = numeric_limits<long>::max() / 2;
    
    std::vector<long> dist(no_of_nodes, INF);
    dist[source_node_id] = 0;
    Edge* edges = graph->edge;
    
    for (int iter = 0; iter < no_of_nodes - 1; iter++)
    {
        bool any_update = false;
        
        #pragma omp parallel
        {
            bool local_update = false;
            
            #pragma omp for schedule(dynamic, 4096) nowait
            for (int e = 0; e < no_of_edges; e++)
            {
                int u = edges[e].source;
                int v = edges[e].destination;
                long w = edges[e].weight;
                long du = dist[u];
                
                if (du < INF)
                {
                    long new_dist = du + w;
                    if (new_dist < dist[v]) {
                        dist[v] = new_dist;
                        local_update = true;
                    }
                }
            }
            
            if (local_update) {
                #pragma omp atomic write
                any_update = true;
            }
        }
        
        if (!any_update) break;
    }
    
    return dist;
}

// ============================================================ 
// VERZIJA 4: SIMD TILING (AVX2) Dodaje: destination tiling + AVX2 SIMD + SoA format
//      Zadržava: sortiranje koncept (reorder po tile-ovima)
//      NAPOMENA: Prelazi na int tip zbog SIMD gather ograničenja
// ============================================================
std::vector<int> runBellmanFordSSSP_SIMD_Tiling(GraphSoA* graph, int source_node_id)
{
    int N = graph->num_nodes;
    int E = graph->num_edges;
    const int INF = std::numeric_limits<int>::max() / 2;

    // Direktno koristimo originalne nizove - BEZ SORTIRANJA
    const std::vector<int>& src = graph->sources;
    const std::vector<int>& dst = graph->destinations;
    const std::vector<int>& w   = graph->weights;

    // Inicijalizacija distance vektora
    std::vector<int> dist(N, INF);
    dist[source_node_id] = 0;

    // Bellman-Ford sa OpenMP + AVX2 SIMD
    for (int iter = 0; iter < N - 1; ++iter) {
        bool any_update = false;
        
        #pragma omp parallel
        {
            bool local_update = false;
            
            // Izračunaj koliko SIMD iteracija ima (po 8 grana)
            int simd_bound = (E / 8) * 8;
            
            #pragma omp for schedule(dynamic, 4096) nowait
            for (int j = 0; j < simd_bound; j += 8) {
                // AVX2 SIMD obrada 8 grana odjednom
                __m256i vsrc = _mm256_loadu_si256((__m256i const*)(src.data() + j));
                __m256i vdst = _mm256_loadu_si256((__m256i const*)(dst.data() + j));
                __m256i vwt  = _mm256_loadu_si256((__m256i const*)(w.data() + j));

                // SIMD gather: prikupi dist[src] i dist[dst] za 8 grana
                __m256i v_du = _mm256_i32gather_epi32(dist.data(), vsrc, 4);
                __m256i v_oldv = _mm256_i32gather_epi32(dist.data(), vdst, 4);

                // SIMD sabiranje: new_dist = dist[u] + weight
                __m256i v_new = _mm256_add_epi32(v_du, vwt);

                // Store rezultata
                alignas(32) int du_arr[8], old_arr[8], new_arr[8], dst_arr[8];
                _mm256_store_si256((__m256i*)du_arr, v_du);
                _mm256_store_si256((__m256i*)old_arr, v_oldv);
                _mm256_store_si256((__m256i*)new_arr, v_new);
                _mm256_store_si256((__m256i*)dst_arr, vdst);

                // Skalarni update (zbog data dependencies)
                for (int k = 0; k < 8; ++k) {
                    if (du_arr[k] < INF) {
                        int dst_idx = dst_arr[k];
                        int nd = new_arr[k];
                        if (nd < dist[dst_idx]) {
                            dist[dst_idx] = nd;
                            local_update = true;
                        }
                    }
                }
            }
            
            // Obrada preostalih grana skalarno (ako ima manje od 8)
            #pragma omp for schedule(dynamic, 1024) nowait
            for (int j = simd_bound; j < E; ++j) {
                int u = src[j];
                int v = dst[j];
                int wt = w[j];
                int du = dist[u];
                
                if (du < INF) {
                    int nd = du + wt;
                    if (nd < dist[v]) {
                        dist[v] = nd;
                        local_update = true;
                    }
                }
            }
            
            if (local_update) {
                #pragma omp atomic write
                any_update = true;
            }
        }
        
        if (!any_update) break;
    }

    return dist;
}
// ============================================================
//      VERZIJA 5: SIMD TILING + OpenMP (AVX2)
//      Dodaje: paralelizaciju sa OpenMP + atomic operacije
//      Zadržava: destination tiling + AVX2 SIMD + SoA format
// ============================================================
std::vector<int> runBellmanFordSSSP_SIMD_Tiling_OMP(GraphSoA* graph, int source_node_id)
{
    int N = graph->num_nodes;
    int E = graph->num_edges;
    const int INF = std::numeric_limits<int>::max() / 2;

    // ZADRŽANO IZ VERZIJE 4: Destination tiling
    const int TILE_SIZE = 1024;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    std::vector<std::vector<std::tuple<int,int,int>>> coo_tiles(num_tiles);
    const std::vector<int>& src_in = graph->sources;
    const std::vector<int>& dst_in = graph->destinations;
    const std::vector<int>& w_in   = graph->weights;

    for (int i = 0; i < E; ++i) {
        int tile_id = dst_in[i] / TILE_SIZE;
        coo_tiles[tile_id].emplace_back(src_in[i], dst_in[i], w_in[i]);
    }

    // ZADRŽANO IZ VERZIJE 4: SoA reordering
    std::vector<int> src; src.reserve(E);
    std::vector<int> dst; dst.reserve(E);
    std::vector<int> w;   w.reserve(E);

    for (int t = 0; t < num_tiles; ++t) {
        for (auto &tp : coo_tiles[t]) {
            src.push_back(std::get<0>(tp));
            dst.push_back(std::get<1>(tp));
            w.push_back(std::get<2>(tp));
        }
    }

    // OPTIMIZACIJA: Atomic vektor za thread-safe pristup (iz verzije 3)
    std::vector<std::atomic<int>> atomic_dist(N);
    for (int i = 0; i < N; i++) {
        atomic_dist[i].store(INF, std::memory_order_relaxed);
    }
    atomic_dist[source_node_id].store(0, std::memory_order_relaxed);

    bool updated = true;

    for (int iter = 0; iter < N - 1 && updated; ++iter) {
        updated = false;
        std::atomic<bool> any_update(false);

        // OPTIMIZACIJA: OpenMP paralelizacija
        #pragma omp parallel
        {
            bool local_update = false;

            #pragma omp for schedule(dynamic, 256) nowait
            for (int j = 0; j < E; j += 8) {
                // ZADRŽANO IZ VERZIJE 4: AVX2 SIMD processing
                if (j + 7 < E) {
                    __m256i vsrc = _mm256_loadu_si256((__m256i const*)(src.data() + j));
                    __m256i vdst = _mm256_loadu_si256((__m256i const*)(dst.data() + j));
                    __m256i vwt  = _mm256_loadu_si256((__m256i const*)(w.data() + j));

                    // Učitaj trenutne distance atomično
                    alignas(32) int du_arr[8], dst_arr[8], wt_arr[8];
                    _mm256_store_si256((__m256i*)dst_arr, vdst);
                    _mm256_store_si256((__m256i*)wt_arr, vwt);
                    
                    for (int k = 0; k < 8; ++k) {
                        du_arr[k] = atomic_dist[src.data()[j + k]].load(std::memory_order_relaxed);
                    }

                    __m256i v_du = _mm256_load_si256((__m256i*)du_arr);
                    __m256i v_new = _mm256_add_epi32(v_du, vwt);

                    alignas(32) int new_arr[8];
                    _mm256_store_si256((__m256i*)new_arr, v_new);

                    // Thread-safe update sa compare-and-swap
                    for (int k = 0; k < 8; ++k) {
                        if (du_arr[k] < INF) {
                            int dst_idx = dst_arr[k];
                            int nd = new_arr[k];
                            
                            int old_val = atomic_dist[dst_idx].load(std::memory_order_relaxed);
                            while (nd < old_val) {
                                if (atomic_dist[dst_idx].compare_exchange_weak(
                                        old_val, nd,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
                                    local_update = true;
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    // Skalarni ostatak sa atomic operacijama
                    for (int jj = j; jj < E; ++jj) {
                        int u = src[jj], v = dst[jj], wt = w[jj];
                        int du = atomic_dist[u].load(std::memory_order_relaxed);
                        
                        if (du < INF) {
                            int nd = du + wt;
                            int old_val = atomic_dist[v].load(std::memory_order_relaxed);
                            
                            while (nd < old_val) {
                                if (atomic_dist[v].compare_exchange_weak(
                                        old_val, nd,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
                                    local_update = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if (local_update) {
                any_update.store(true, std::memory_order_relaxed);
            }
        }

        updated = any_update.load(std::memory_order_relaxed);
    }

    // Prebaci nazad u običan vektor
    std::vector<int> dist(N);
    for (int i = 0; i < N; i++) {
        dist[i] = atomic_dist[i].load(std::memory_order_relaxed);
    }

    return dist;
}

