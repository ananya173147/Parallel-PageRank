#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;


void get_adj_matrix(vector<vector<double>>& graph, int& n, double d, FILE* inputFilePtr, bool is_mtx = false) {
    if (!inputFilePtr) {
        cerr << "Failed to open input file." << endl;
        exit(1);
    }

    int m, indexing = 1;

    if (is_mtx) {
        char line[512];
        while (fgets(line, sizeof(line), inputFilePtr)) {
            if (line[0] != '%') {
                sscanf(line, "%d %d %d", &n, &n, &m); // Read matrix size (n x n) and edge count
                break;
            }
        }
    } else {
        if (fscanf(inputFilePtr, "%d%d", &m, &indexing) != 2) {
            cerr << "Invalid edge count or indexing type." << endl;
            exit(1);
        }
    }

    graph.assign(n, vector<double>(n, (1 - d) / n));

    while (true) {
        int src, dst;
        if (fscanf(inputFilePtr, "%d%d", &src, &dst) != 2)
            break;

        if (indexing == 0 || !is_mtx)
            graph[dst][src] += d;
        else
            graph[dst - 1][src - 1] += d;
    }
}


// void get_adj_matrix(vector<vector<double>>& graph, int n, double d, FILE* inputFilePtr) {
//     if (!inputFilePtr) {
//         cerr << "Failed to open input file." << endl;
//         exit(1);
//     }

//     int m, indexing;
//     if (fscanf(inputFilePtr, "%d%d", &m, &indexing) != 2) {
//         cerr << "Invalid edge count or indexing type." << endl;
//         exit(1);
//     }

//     // Initialize graph with teleportation base value
//     for (int i = 0; i < n; ++i)
//         fill(graph[i].begin(), graph[i].end(), (1 - d) / n);

//     while (m--) {
//         int src, dst;
//         if (fscanf(inputFilePtr, "%d%d", &src, &dst) != 2) {
//             cerr << "Invalid edge line." << endl;
//             exit(1);
//         }
//         if (indexing == 0)
//             graph[dst][src] += d;
//         else
//             graph[dst - 1][src - 1] += d;
//     }
// }

void manage_adj_matrix(vector<vector<double>>& graph, int n) {
    for (int j = 0; j < n; ++j) {
        double col_sum = 0.0;
        for (int i = 0; i < n; ++i)
            col_sum += graph[i][j];

        if (col_sum != 0) {
            for (int i = 0; i < n; ++i)
                graph[i][j] /= col_sum;
        } else {
            for (int i = 0; i < n; ++i)
                graph[i][j] = 1.0 / n;
        }
    }
}

void power_method(const vector<vector<double>>& graph, vector<double>& r, int n, int max_iter = 1000, double eps = 1e-6) {
    vector<double> r_last(n, 1.0 / n);
    r.assign(n, 0.0);

    while (max_iter--) {
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j)
                sum += r_last[j] * graph[i][j];
            r[i] = sum;
        }

        // Check convergence
        double diff = 0.0;
        for (int i = 0; i < n; ++i) {
            diff += abs(r[i] - r_last[i]);
            r_last[i] = r[i];
        }

        if (diff < eps)
            break;
    }
}

void top_nodes(const vector<double>& r, int count = 10) {
    vector<pair<double, int>> ranked;
    for (int i = 0; i < r.size(); ++i)
        ranked.emplace_back(r[i], i + 1); // 1-based index

    partial_sort(ranked.begin(), ranked.begin() + count, ranked.end(), greater<>());

    for (int i = 0; i < count; ++i)
        printf("Rank %d Node is %d (Score: %.6f)\n", i + 1, ranked[i].second, ranked[i].first);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: ./pagerank_openmp <inputfile> <num_runs>" << endl;
        return 1;
    }

    const char* filename = argv[1];
    int NUM_RUNS = atoi(argv[2]);

    FILE* inputFilePtr = fopen(filename, "r");
    if (!inputFilePtr) {
        cerr << "Cannot open file " << filename << endl;
        return 1;
    }

    // int n;
    // if (fscanf(inputFilePtr, "%d", &n) != 1) {
    //     cerr << "Error reading number of nodes." << endl;
    //     return 1;
    // }
    int n=0;

    double d = 0.85;
    vector<vector<double>> original_graph(n, vector<double>(n));
    // get_adj_matrix(original_graph, n, d, inputFilePtr);

    string filename = argv[1];
    bool is_mtx = filename.size() > 4 && filename.substr(filename.size() - 4) == ".mtx";

    get_adj_matrix(graph, n, d, inputFilePtr, is_mtx);

    fclose(inputFilePtr);

    vector<double> r(n);
    double total_ms = 0.0;

    for (int run = 0; run < NUM_RUNS; ++run) {
        // Deep copy of graph to avoid mutation between runs
        vector<vector<double>> graph = original_graph;

        auto start = high_resolution_clock::now();
        manage_adj_matrix(graph, n);
        power_method(graph, r, n);
        auto end = high_resolution_clock::now();

        total_ms += duration_cast<milliseconds>(end - start).count();
    }

    // Print results from last run
    top_nodes(r);

    cout << fixed << setprecision(2);
    cout << "Average Time taken: " << (total_ms / NUM_RUNS) << " ms over " << NUM_RUNS
         << " runs for " << n << " nodes." << endl;

    return 0;
}
