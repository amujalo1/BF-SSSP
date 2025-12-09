@echo off
echo ================================
echo     BUILD SCRIPT FOR BF-SSSP
echo ================================
echo.

REM -------- COMMON COMPILER FLAGS --------
set FLAGS=-std=c++20 -O3 -march=native -mavx512f -mavx512dq -mavx512vl -mtune=native -fopenmp -flto -funroll-loops -fno-signed-zeros -freciprocal-math -ffast-math

REM =======================================
REM ============   V1 SERIES   ============
REM =======================================

echo Building V1r...
g++ %FLAGS% src/main_V1r.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V1r.exe

echo Building V1n...
g++ %FLAGS% src/main_V1n.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V1n.exe

echo Building V1d...
g++ %FLAGS% src/main_V1d.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V1d.exe

REM =======================================
REM ============   V2 SERIES   ============
REM =======================================

echo Building V2r...
g++ %FLAGS% src/main_V2r.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V2r.exe

echo Building V2n...
g++ %FLAGS% src/main_V2n.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V2n.exe

echo Building V2d...
g++ %FLAGS% src/main_V2d.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V2d.exe

REM =======================================
REM ============   V3 SERIES   ============
REM =======================================

echo Building V3r...
g++ %FLAGS% src/main_V3r.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V3r.exe

echo Building V3n...
g++ %FLAGS% src/main_V3n.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V3n.exe

echo Building V3d...
g++ %FLAGS% src/main_V3d.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V3d.exe

REM =======================================
REM ============   V4 SERIES   ============
REM =======================================

echo Building V4r...
g++ %FLAGS% src/main_V4r.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V4r.exe

echo Building V4n...
g++ %FLAGS% src/main_V4n.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V4n.exe

echo Building V4d...
g++ %FLAGS% src/main_V4d.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V4d.exe

REM =======================================
REM ============   V5 SERIES   ============
REM =======================================

echo Building V5r...
g++ %FLAGS% src/main_V5r.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V5r.exe

echo Building V5n...
g++ %FLAGS% src/main_V5n.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V5n.exe

echo Building V5d...
g++ %FLAGS% src/main_V5d.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V5d.exe

REM =======================================
REM ============   V6 ONLY d   ============
REM =======================================

echo Building V6d...
g++ %FLAGS% src/main_V6d.cpp src/graph_utils.cpp src/bellman_ford.cpp -I incl -o V6d.exe


echo.
echo ================================
echo        BUILD COMPLETE
echo ================================
echo.