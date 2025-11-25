# Bellman-Ford SSSP Optimizacija (Cache, OpenMP, SIMD AVX/AVX-512)

Ovaj repozitorij sadrži implementaciju i postepenu optimizaciju algoritma Bellman-Ford za problem najkraćeg puta iz jednog izvora (Single-Source Shortest Path - SSSP) na rijetkim grafovima. Primarni cilj bio je istražiti efekte optimizacija za memorijsku hijerarhiju (Cache), multi-threading (OpenMP) i vektorizaciju (SIMD AVX2/AVX-512).

Testiranje je izvršeno na grafu sa **200,000 čvorova** i **8,000,000 grana**.

## 🚀 Postepena Progresija Optimizacije

Implementirano je sedam verzija algoritma koje postepeno grade optimizacije.

| Verzija | Tehnika | Detalji Implementacije | Format Grafa |
| :---: | :--- | :--- | :--- |
| **V1** | **Originalna (Baseline)** | Standardna iterativna implementacija. | AoS (Array of Structs) |
| **V2** | **CACHE Optim.** | Sortiranje grana po izvornom čvoru (Source Node), implementacija Blockinga i eksplicitni prefetch. | AoS (Array of Structs) |
| **V3** | **OpenMP** | Paralelizacija vanjske petlje (relaksacije) koristeći OpenMP i `std::atomic` za thread-safe update distanci. | AoS (Array of Structs) |
| **V4** | **SIMD Tiling (AVX2)** | Prelazak na SoA format, implementacija Destination-based Tilinga, te relaksacija grana koristeći **AVX2** (8-way SIMD). | SoA (Structure of Arrays) |
| **V5** | **V4 + OpenMP (AVX2)** | Kombinacija SIMD vektorizacije (V4) i OpenMP multi-threadinga. Trenutno najbrža verzija. | SoA (Structure of Arrays) |
| **V6** | **SIMD Tiling (AVX-512)** | Proširenje V4 verzije korištenjem **AVX-512** (16-way SIMD) instrukcija. | SoA (Structure of Arrays) |
| **V7** | **V6 + OpenMP (AVX-512)** | Kombinacija AVX-512 SIMD i OpenMP paralelizacije. | SoA (Structure of Arrays) |

---

## 📊 Rezultati Mjerenja (Vrijeme i Ubrzanje)

Vrijeme je mjereno u sekundama. Ubrzanje (`Speedup`) se računa u odnosu na **Originalnu verziju (V1)**:
$$\text{Ubrzanje} = \frac{\text{Vrijeme(V1)}}{\text{Vrijeme(Vx)}}$$

### Finalni Rezime

| Verzija | Tehnika | Vrijeme (s) | Ubrzanje (x) | Korektnost |
|:--------|:--------------------------------|:------------|:-------------|:-----------|
| **V1** | Originalna (Baseline)           | **1.124** | N/A          | N/A        |
| **V2** | Cache Optim. (Sort+Block+Pref)  | 0.557       | 2.02         | V TACNO    |
| **V3** | OpenMP (Multi-threading)        | 0.506       | 2.22         | V TACNO    |
| **V4** | SIMD Tiling (AVX2, 8-way)       | 0.408       | 2.75         | V TACNO    |
| **V5** | **V4 + OpenMP (AVX2)** | **0.184** | **6.10** | V TACNO    |
| **V6** | SIMD Tiling (AVX-512, 16-way)   | 0.409       | 2.75         | V TACNO    |
| **V7** | V6 + OpenMP (ULTIMATIVNA)       | 0.212       | 5.30         | V TACNO    |

### Analiza Ključnih Rezultata

1.  **Cache Optimizacija (V2):** Sama implementacija sortiranja grana i blockinga donosi značajno poboljšanje, skraćujući vrijeme sa **1.124s na 0.557s (2.02x)**. Ovo potvrđuje da je Bellman-Ford na rijetkim grafovima I/O bound (ograničen memorijskim pristupom).
2.  **Kombinacija SIMD i OpenMP (V5):** Verzija **V5** (SIMD AVX2 + OpenMP) je **najbrža implementacija**, postižući ubrzanje od **6.10x** u odnosu na baseline. Kombinacija lokalnosti SoA formata, paralelizacije niti i vektorizacije relaksacija je bila ključna.
3.  **AVX-512 Efekat (V6 & V7):**
    * Samo AVX-512 (V6) ne donosi nikakvo poboljšanje u odnosu na AVX2 (V4), zadržavajući ubrzanje od **2.75x**.
    * Verzija V7 (AVX-512 + OpenMP) je sporija od V5 (AVX2 + OpenMP). Ovo je uobičajen rezultat na modernim CPU arhitekturama zbog tzv. **AVX-512 frekvencijskog skaliranja (down-clocking)**. Iako V7 koristi šire registre, niža radna frekvencija CPU-a tokom AVX-512 izvršavanja poništava benefite vektorizacije.

**Zaključak:** Najbolja strategija za performanse u ovom slučaju bila je **kombinovanje AVX2 SIMD-a i OpenMP paralelizacije (V5)**, izbjegavajući skuplji overhead AVX-512 instrukcija.

---

## 🛠️ Kompilacija i Izvršavanje

Projekat je kompajliran koristeći `g++` sa agresivnim optimizacijama za performanse (`-O3`, `-ffast-math`) i ciljanjem specifičnih CPU instrukcija (`-march=native`, `-mavx`, `-mavx512f`).

### Potrebne Zastavice

Sve kompilacije zahtijevaju linkovanje sa OpenMP bibliotekom (`-fopenmp`).

```bash
# Za AVX-512 verzije (V6, V7)
g++ -std=c++20 -O3 -march=native -mavx512f -mavx512dq -mavx512vl -mtune=native -fopenmp ...

# Za AVX2 verzije (V4, V5)
g++ -std=c++20 -O3 -march=native -mavx -mavx2 -fopenmp ...