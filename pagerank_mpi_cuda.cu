#include <mpi.h>
#include <stdio.h>
#include <cuda.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <bits/stdc++.h>
using namespace std;

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, err); \
    } \
} while(0)

#define MPI_CHECK(call) \
do { \
    int err = call; \
    if (err != MPI_SUCCESS) { \
        char error_string[MPI_MAX_ERROR_STRING]; \
        int length; \
        MPI_Error_string(err, error_string, &length); \
        fprintf(stderr, "MPI error in %s:%d: %s\n", __FILE__, __LINE__, error_string); \
        MPI_Abort(MPI_COMM_WORLD, err); \
    } \
} while(0)

__global__ void manage_adj_matrix(float* gpu_graph, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){
        float sum = 0.0;
        for (int i = 0; i < n; ++i){
            sum += gpu_graph[i * n + id];
        }
        for (int i = 0; i < n; ++i){
            if (sum != 0.0){
                gpu_graph[i * n + id] /= sum;
            }
            else{
                gpu_graph[i * n + id] = 1.0 / n;
            }
        }
    }
}

__global__ void initialize_rank(float* gpu_r, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){
        gpu_r[id] = 1.0f / n;
    }
}

__global__ void store_rank(float* gpu_r, float* gpu_r_last, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){
        gpu_r_last[id] = gpu_r[id];
    }
}

__global__ void matmul(float* gpu_graph, float* gpu_r, float* gpu_r_last, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){
        float sum = 0.0;
        for (int j = 0; j < n; ++j){
            sum += gpu_r_last[j] * gpu_graph[id * n + j];
        }
        gpu_r[id] = sum;
    }
}

__global__ void rank_diff(float* gpu_r, float* gpu_r_last, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){
        gpu_r_last[id] = fabsf(gpu_r_last[id] - gpu_r[id]);
    }
}

__global__ void init_pair_array(float* gpu_r, float* values, int* indices, int n){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){
        values[id] = gpu_r[id];
        indices[id] = id + 1; // 1-based
    }
}

void get_adj_matrix(float* graph, int n, float d, FILE *inputFilePtr ){
    int m, indexing;
    fscanf(inputFilePtr, "%d", &m);
    fscanf(inputFilePtr, "%d", &indexing);

    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; ++j)
            graph[i * n + j] = (1 - d) / n;

    while(m--){
        int src, dst;
        fscanf(inputFilePtr, "%d%d", &src, &dst);
        if(indexing == 0)
            graph[dst * n + src] += d;
        else
            graph[(dst - 1) * n + (src - 1)] += d;
    }
}

int main(int argc, char** argv) {

    MPI_CHECK(MPI_Init(&argc, &argv));
    
    int world_size, world_rank;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    MPI_Comm node_comm;
    int local_rank, local_size;

    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm));
    MPI_CHECK(MPI_Comm_rank(node_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(node_comm, &local_size));

    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

    if (num_gpus == 0) {
        if (world_rank == 0) printf("No GPUs found on node. Exiting.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    int assigned_gpu = local_rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(assigned_gpu));

    if (world_rank == 0) {
        printf("Multi-GPU setup: %d MPI ranks per node, %d GPUs per node.\n", local_size, num_gpus);
    }
    printf("Rank %d using GPU %d\n", world_rank, assigned_gpu);

    if (argc < 4) {
        if (world_rank == 0)
            printf("Usage: mpirun -np <numprocs> ./pagerank_mpi input.txt BLOCKSIZE NUM_RUNS\n");
        MPI_Finalize();
        return 1;
    }

    char* inputfile = argv[1];
    int BLOCKSIZE = atoi(argv[2]);
    int NUM_RUNS = atoi(argv[3]);

    FILE* inputFilePtr = fopen(inputfile, "r");
    if (!inputFilePtr) {
        if (world_rank == 0) printf("Cannot open input file.\n");
        MPI_Finalize();
        return 1;
    }

    int n;
    fscanf(inputFilePtr, "%d", &n);
    int nblocks = ceil((float)n / BLOCKSIZE);

    float* graph = (float*) malloc(n * n * sizeof(float));
    float* r = (float*) malloc(n * sizeof(float));
    float d = 0.85;

    get_adj_matrix(graph, n, d, inputFilePtr);
    fclose(inputFilePtr);


    // Allocate GPU memory
    float *gpu_graph, *gpu_r, *gpu_r_last;
    cudaMalloc(&gpu_graph, n * n * sizeof(float));
    cudaMemcpy(gpu_graph, graph, n * n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_r, n * sizeof(float));
    cudaMalloc(&gpu_r_last, n * sizeof(float));
    initialize_rank<<<nblocks, BLOCKSIZE>>>(gpu_r, n);
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    manage_adj_matrix<<<nblocks, BLOCKSIZE>>>(gpu_graph, n);
    cudaDeviceSynchronize();

    int max_iter = 5000;
    float eps = 1e-6;

    for (int iter = 0; iter < max_iter; ++iter) {
        store_rank<<<nblocks, BLOCKSIZE>>>(gpu_r, gpu_r_last, n);
        cudaDeviceSynchronize();

        matmul<<<nblocks, BLOCKSIZE>>>(gpu_graph, gpu_r, gpu_r_last, n);
        cudaDeviceSynchronize();

        rank_diff<<<nblocks, BLOCKSIZE>>>(gpu_r, gpu_r_last, n);
        cudaDeviceSynchronize();

        float local_diff = thrust::reduce(thrust::device, gpu_r_last, gpu_r_last + n);
        float global_diff;

        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        if (global_diff < eps) {
            if (world_rank == 0) printf("Converged at iteration %d\n", iter);
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (world_rank == 0) {
        printf("Total time: %.3f ms\n", 1000*(end_time - start_time));

        cudaMemcpy(r, gpu_r, n * sizeof(float), cudaMemcpyDeviceToHost);
        vector<pair<float, int>> ranking;
        for (int i = 0; i < n; ++i)
            ranking.push_back({r[i], i+1});
        sort(ranking.rbegin(), ranking.rend());

        for (int i = 0; i < min(10, n); ++i)
            printf("Rank %d Node: %d (Score: %.6f)\n", i+1, ranking[i].second, ranking[i].first);
    }

    cudaFree(gpu_graph);
    cudaFree(gpu_r);
    cudaFree(gpu_r_last);
    free(graph);
    free(r);

    MPI_Finalize();
    return 0;
}