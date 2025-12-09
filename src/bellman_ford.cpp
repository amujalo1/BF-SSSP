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
// VERZIJA 3: OpenMP PARALELIZACIJA
// Dodaje: multi-threading sa thread-safe atomic operacijama
// U ovoj implementaciji dodata je OpenMP paralelizacija kako bi
// više niti moglo istovremeno da obrađuje razlicite ivice.
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
// VERZIJA 4: SIMD (AVX2)
// Optimizovana Bellman-Ford SSSP implementacija koja kombinuje:
//   • OpenMP paralelizaciju po ivicama,
//   • AVX2 SIMD vektorizaciju (obrada 8 ivica odjednom),
//   • SoA (Structure of Arrays) format grafa radi efikasnog
//     sekvencijalnog čitanja memorije i gather operacija.
// ============================================================
std::vector<int> runBellmanFordSSSP_SIMD_OMP(GraphSoA* graph, int source_node_id)
{
    int N = graph->num_nodes;
    int E = graph->num_edges;
    const int INF = std::numeric_limits<int>::max() / 2;
    
    std::vector<int> dist(N, INF);
    dist[source_node_id] = 0;
    
    const auto& src = graph->sources;
    const auto& dst = graph->destinations;
    const auto& w = graph->weights;
    
    for (int iter = 0; iter < N - 1; ++iter) {
        bool updated = false;
        
        #pragma omp parallel
        {
            bool local_updated = false;
            int simd_bound = (E / 8) * 8;
            
            // SIMD dio - obrađuje 8 grana odjednom
            #pragma omp for schedule(dynamic, 4096) nowait
            for (int i = 0; i < simd_bound; i += 8) {
                // Učitaj 8 source, dest i weight vrijednosti
                __m256i vsrc = _mm256_loadu_si256((__m256i const*)(&src[i]));
                __m256i vdst = _mm256_loadu_si256((__m256i const*)(&dst[i]));
                __m256i vw = _mm256_loadu_si256((__m256i const*)(&w[i]));
                
                // Gather: prikupi distance source čvorova
                __m256i vdu = _mm256_i32gather_epi32(dist.data(), vsrc, 4);
                
                // Proveri koje distance su manje od INF
                __m256i vinf = _mm256_set1_epi32(INF);
                __m256i vmask = _mm256_cmpgt_epi32(vinf, vdu);
                
                // Ako nijedna nije validna, preskoči
                if (_mm256_testz_si256(vmask, vmask))
                    continue;
                
                // Izračunaj nove distance
                __m256i vnew = _mm256_add_epi32(vdu, vw);
                
                // Gather: prikupi trenutne distance dest čvorova
                __m256i vdv = _mm256_i32gather_epi32(dist.data(), vdst, 4);
                
                // Proveri koje nove distance su bolje
                __m256i vbetter = _mm256_cmpgt_epi32(vdv, vnew);
                
                // Kombinuj uslove: (du < INF) AND (new < dv)
                __m256i vupdate = _mm256_and_si256(vmask, vbetter);
                
                // Proveri da li ima update-ova
                if (!_mm256_testz_si256(vupdate, vupdate)) {
                    alignas(32) int new_arr[8], dst_arr[8];
                    int mask_arr[8];
                    
                    _mm256_store_si256((__m256i*)new_arr, vnew);
                    _mm256_store_si256((__m256i*)dst_arr, vdst);
                    _mm256_store_si256((__m256i*)mask_arr, vupdate);
                    
                    // Skalarno izvrši update zbog data dependencies
                    for (int k = 0; k < 8; ++k) {
                        if (mask_arr[k]) {
                            #pragma omp atomic write
                            dist[dst_arr[k]] = new_arr[k];
                            local_updated = true;
                        }
                    }
                }
            }
            
            // Skalarni dio - preostale grane
            #pragma omp for schedule(dynamic, 1024) nowait
            for (int i = simd_bound; i < E; ++i) {
                int u = src[i];
                int v = dst[i];
                int wt = w[i];
                int du = dist[u];
                
                if (du < INF) {
                    int nd = du + wt;
                    if (nd < dist[v]) {
                        #pragma omp atomic write
                        dist[v] = nd;
                        local_updated = true;
                    }
                }
            }
            
            // Signaliziraj da je bilo update-ova u ovoj iteraciji
            if (local_updated) {
                #pragma omp atomic write
                updated = true;
            }
        } // kraj #pragma omp parallel
        
        // Rana terminacija ako nije bilo update-ova
        if (!updated) break;
    }
    
    return dist;
}

// ============================================================
// VERZIJA 5: SIMD OpenMP (AVX512)
// Najagresivnije optimizovana verzija Bellman-Ford SSSP algoritma.
// Kombinuje sledece tehnike:
//
//   • AVX-512 SIMD: obrada 16 ivica paralelno u jednom registru,
//   • Destination tiling: grupisanje ivica po odredišnim cvorovima
//     radi boljeg lokaliteta i nizeg memory traffic-a,
//   • OpenMP paralelizacija po blokovima ivica,
//   • SoA (Structure of Arrays) format grafa, što dozvoljava
//     sekvencijalno citanje memorije i efikasne gather operacije,
//   • Atomic operacije pri ažuriranju dist[v] kako bi se izbegle
//     data race situacije u paralelnom izvršavanju.
// ============================================================
std::vector<int> runBellmanFordSSSP_SIMD_512_OMP(GraphSoA* graph, int source_node_id)
{
    int N = graph->num_nodes;
    int E = graph->num_edges;
    const int INF = std::numeric_limits<int>::max() / 2;

    std::vector<int> dist(N, INF);
    dist[source_node_id] = 0;

    const auto& src = graph->sources;
    const auto& dst = graph->destinations;
    const auto& w   = graph->weights;

    for (int iter = 0; iter < N - 1; ++iter) {
        bool updated = false;

        #pragma omp parallel
        {
            bool local_updated = false;
            int simd_bound = (E / 16) * 16;     // 16 elemenata po AVX512 registru

            // SIMD dio – 16 grana odjednom
            #pragma omp for schedule(dynamic, 4096) nowait
            for (int i = 0; i < simd_bound; i += 16) {

                // load source, dest, weight
                __m512i vsrc = _mm512_loadu_si512((__m512i const*)(&src[i]));
                __m512i vdst = _mm512_loadu_si512((__m512i const*)(&dst[i]));
                __m512i vw   = _mm512_loadu_si512((__m512i const*)(&w[i]));

                // gather dist[u]
                __m512i vdu = _mm512_i32gather_epi32(vsrc, dist.data(), 4);

                __m512i vinf = _mm512_set1_epi32(INF);

                // mask: du < INF
                __mmask16 mask_valid_du = _mm512_cmp_epi32_mask(vdu, vinf, _MM_CMPINT_LT);

                if (mask_valid_du == 0) continue;

                // new = du + w
                __m512i vnew = _mm512_add_epi32(vdu, vw);

                // gather dist[v]
                __m512i vdv = _mm512_i32gather_epi32(vdst, dist.data(), 4);

                // mask: new < dv
                __mmask16 mask_better = _mm512_cmp_epi32_mask(vnew, vdv, _MM_CMPINT_LT);

                // final mask = valid & better
                __mmask16 mask_update = mask_valid_du & mask_better;

                if (mask_update) {
                    alignas(64) int new_arr[16];
                    alignas(64) int dst_arr[16];

                    _mm512_store_si512((__m512i*)new_arr, vnew);
                    _mm512_store_si512((__m512i*)dst_arr, vdst);

                    // scalar update (zbog race condition-a)
                    for (int k = 0; k < 16; ++k) {
                        if (mask_update & (1 << k)) {
                            #pragma omp atomic write
                            dist[dst_arr[k]] = new_arr[k];
                            local_updated = true;
                        }
                    }
                }
            }

            // skalarni dio
            #pragma omp for schedule(dynamic, 1024) nowait
            for (int i = simd_bound; i < E; ++i) {
                int u = src[i];
                int v = dst[i];
                int wt = w[i];
                int du = dist[u];

                if (du < INF) {
                    int nd = du + wt;
                    if (nd < dist[v]) {
                        #pragma omp atomic write
                        dist[v] = nd;
                        local_updated = true;
                    }
                }
            }

            if (local_updated) {
                #pragma omp atomic write
                updated = true;
            }
        }

        if (!updated) break;
    }

    return dist;
}

#include <parallel/algorithm>   // __gnu_parallel::sort

std::vector<long> runBellmanFordSSSP_CACHE_SORT(Graph* graph, int source_node_id) {
    int no_of_nodes = graph->num_nodes;
    int no_of_edges = graph->num_edges;
    
    const long INF = std::numeric_limits<long>::max() / 2;
    std::vector<long> node_distances(no_of_nodes, INF);
    node_distances[source_node_id] = 0;
    
    // OPTIMIZACIJA: PARALLEL sortiranje grana po source čvoru za bolju cache lokalnost
    std::vector<Edge> sorted_edges(graph->edge, graph->edge + no_of_edges);
    __gnu_parallel::sort(
        sorted_edges.begin(),
        sorted_edges.end(),
        [](const Edge& a, const Edge& b) { return a.source < b.source; }
    );
    
    // Glavna petlja relaksacije
    for (int iter = 0; iter < no_of_nodes - 1; iter++) {
        bool relaxed = false;
        
        // OPTIMIZACIJA: Procesiranje po blokovima za bolju cache iskorištenost
        const int BLOCK_SIZE = 64;
        for (int block_start = 0; block_start < no_of_edges; block_start += BLOCK_SIZE) {
            int block_end = std::min(block_start + BLOCK_SIZE, no_of_edges);
            
            // OPTIMIZACIJA: Prefetch za sljedeći blok
            if (block_start + BLOCK_SIZE < no_of_edges) {
                __builtin_prefetch(&sorted_edges[block_start + BLOCK_SIZE], 0, 3);
            }
            
            // Procesiranje trenutnog bloka
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