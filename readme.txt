This project implements the PageRank algorithm using multiple parallel programming paradigms:

1. sequential.cpp         → Basic single-threaded CPU implementation
2. op.cpp                 → Multi-threaded CPU implementation using OpenMP
3. cuda.cu                → GPU-accelerated version using CUDA
4. pagerank_mpi_cuda.cu   → Distributed GPU version using MPI + CUDA hybrid

Build Instructions
------------------
Make sure required modules or tools are available:
- g++ (with OpenMP support)
- NVIDIA CUDA Toolkit
- MPI (e.g., OpenMPI or MPICH)

To build all implementations:
    make

To clean binaries:
    make clean

Run Instructions
----------------
All implementations expect a graph input file and number of runs (where applicable).
Example graph file formats: custom format or `.mtx` (Matrix Market)

Sequential:
    ./sequential input.txt 3

OpenMP:
    ./op input.txt 3

CUDA:
    ./cuda input.txt 128 3
        # 128 is BLOCKSIZE, 3 is NUM_RUNS

MPI + CUDA (run with mpirun on cluster):
    mpirun -np 2 ./pagerank_mpi_cuda input.txt 128 1
        # 2 MPI ranks, 128 BLOCKSIZE, 1 run

Input Format
------------
Matrix Market (.mtx):
    Header lines starting with '%' are skipped.
    First non-header line: <nrows> <ncols> <nonzeros>
    Following lines: <src> <dst>

Results
-------
Each run prints the top 10 ranked nodes and the average runtime.

Profiling (Optional)
--------------------
CUDA profiling:
    nsys profile --stats=true ./cuda input.txt 128 1

MPI+CUDA profiling:
    nsys profile --stats=true mpirun -np 2 ./pagerank_mpi_cuda input.txt 128 1
