#include <stdio.h>
#include <bits/stdc++.h>
#include <cuda.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

using namespace std;

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
    if (!inputFilePtr) {
        printf("input.txt file failed to open.\n");
        exit(1);
    }

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

void power_method(float* graph, float* r, int n, int nblocks, int BLOCKSIZE, int max_iter = 1000, float eps = 1e-6){
    float* r_last = (float*) malloc(n * sizeof(float));

    float *gpu_graph, *gpu_r, *gpu_r_last;
    cudaMalloc(&gpu_graph, n * n * sizeof(float));
    cudaMemcpy(gpu_graph, graph, n * n * sizeof(float), cudaMemcpyHostToDevice);

    manage_adj_matrix<<<nblocks, BLOCKSIZE>>>(gpu_graph, n);
    cudaDeviceSynchronize();

    cudaMalloc(&gpu_r, n * sizeof(float));
    cudaMalloc(&gpu_r_last, n * sizeof(float));

    initialize_rank<<<nblocks, BLOCKSIZE>>>(gpu_r, n);
    cudaDeviceSynchronize();

    for(int iter = 0; iter < max_iter; ++iter){
        store_rank<<<nblocks, BLOCKSIZE>>>(gpu_r, gpu_r_last, n);
        cudaDeviceSynchronize();

        matmul<<<nblocks, BLOCKSIZE>>>(gpu_graph, gpu_r, gpu_r_last, n);
        cudaDeviceSynchronize();

        rank_diff<<<nblocks, BLOCKSIZE>>>(gpu_r, gpu_r_last, n);
        cudaDeviceSynchronize();

        float result = thrust::reduce(thrust::device, gpu_r_last, gpu_r_last + n);
        if(result < eps){
            printf("Converged at iteration %d\n", iter);
            break;
        }
        
    }

    cudaMemcpy(r, gpu_r, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpu_graph);
    cudaFree(gpu_r);
    cudaFree(gpu_r_last);
    free(r_last);
}

void top_nodes(float* r, int n, int nblocks, int BLOCKSIZE, int count = 10){
    float* gpu_r, *gpu_values;
    int* gpu_indices;
    cudaMalloc(&gpu_r, n * sizeof(float));
    cudaMemcpy(gpu_r, r, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_values, n * sizeof(float));
    cudaMalloc(&gpu_indices, n * sizeof(int));

    init_pair_array<<<nblocks, BLOCKSIZE>>>(gpu_r, gpu_values, gpu_indices, n);
    cudaDeviceSynchronize();

    thrust::sort_by_key(thrust::device, gpu_values, gpu_values + n, gpu_indices);

    vector<float> values(n);
    vector<int> indices(n);
    cudaMemcpy(values.data(), gpu_values, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(indices.data(), gpu_indices, n * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < count; ++i){
        printf("Rank %d Node is %d (Score: %.6f)\n", i + 1, indices[n - 1 - i], values[n - 1 - i]);
    }

    cudaFree(gpu_r);
    cudaFree(gpu_values);
    cudaFree(gpu_indices);
}

int main(int argc, char** argv){
    if(argc < 4){
        printf("Usage: ./pagerank input.txt BLOCKSIZE NUM_RUNS\n");
        return 1;
    }

    char* inputfile = argv[1];
    int BLOCKSIZE = atoi(argv[2]);
    int NUM_RUNS = atoi(argv[3]);

    FILE *inputFilePtr = fopen(inputfile, "r");
    int n;
    fscanf(inputFilePtr, "%d", &n);
    int nblocks = ceil((float)n / BLOCKSIZE);

    float* graph = (float*) malloc(n * n * sizeof(float));
    float* r = (float*) malloc(n * sizeof(float));
    float d = 0.85;

    get_adj_matrix(graph, n, d, inputFilePtr);
    fclose(inputFilePtr);

    float total_time = 0;


    for (int run = 0; run < NUM_RUNS; ++run) {
        float* graph_copy = (float*) malloc(n * n * sizeof(float));
        memcpy(graph_copy, graph, n * n * sizeof(float));

        float* gpu_graph;
        cudaMalloc(&gpu_graph, sizeof(float) * n * n);
        cudaMemcpy(gpu_graph, graph_copy, sizeof(float) * n * n, cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // // GPU work
        // manage_adj_matrix<<<nblocks, BLOCKSIZE>>>(gpu_graph, n);
        // cudaMemcpy(graph_copy, gpu_graph, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
        // cudaFree(gpu_graph);

        power_method(graph_copy, r, n, nblocks, BLOCKSIZE);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        total_time += elapsed_ms;

        // Clean up event objects
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        free(graph_copy);
    }

    top_nodes(r, n, nblocks, BLOCKSIZE);
    printf("Average Time: %.3f ms over for %d nodes.\n", total_time/NUM_RUNS, n);

    free(graph);
    free(r);
    return 0;
}