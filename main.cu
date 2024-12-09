#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// constant definitions 
const int DIMENSIONS = 50;
const int NUM_POINTS = 100000;
const int NUM_QUERIES = 1000;

const int K = 50;
const int BLOCK_SIZE = 256;

// CUDA错误检查实用函数
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// CUDA核函数：计算距离
__device__ double enhancedDistanceGPU(const double* a, const double* b) {
    double sum = 0.0;
    double weight;

    for (int i = 0; i < DIMENSIONS; ++i) {
        weight = 1.0 + sin(i * M_PI / DIMENSIONS);
        double diff = a[i] - b[i];
        sum += weight * (diff * diff);

        if (i % 2 == 0) {
            sum += weight * abs(sin(diff));
        }
        else {
            sum += weight * abs(cos(diff));
        }
    }
    return sqrt(sum);
}

__global__ void computeKNN(const double* data, const double* query,
    double* knnDistances, int* knnIndices, int numPoints) {
    extern __shared__ double sharedDistances[];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算距离，并将结果存入shared memory
    if (globalIdx < numPoints) {
        sharedDistances[tid] = enhancedDistanceGPU(&data[globalIdx * DIMENSIONS], query);
    }
    else {
        sharedDistances[tid] = DBL_MAX;
    }
    __syncthreads();

    // 每个线程负责更新它自己的 K 个最近邻
    if (tid < K) {
        knnDistances[blockIdx.x * K + tid] = sharedDistances[tid];
        knnIndices[blockIdx.x * K + tid] = globalIdx;
    }
}

// 数据生成函数
void generateData(const std::string& filename) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    std::ofstream file(filename);
    for (int i = 0; i < NUM_POINTS; ++i) {
        for (int j = 0; j < DIMENSIONS; ++j) {
            file << dis(gen);
            if (j < DIMENSIONS - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

// 数据加载函数
void loadData(const std::string& filename, std::vector<std::vector<double>>& data) {
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> point;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            point.push_back(std::stod(value));
        }
        data.push_back(point);
    }
    file.close();
}

// 生成查询点
std::vector<std::vector<double>> generateQueryPoints() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    std::vector<std::vector<double>> queryPoints(NUM_QUERIES, std::vector<double>(DIMENSIONS));
    for (int i = 0; i < NUM_QUERIES; ++i) {
        for (int j = 0; j < DIMENSIONS; ++j) {
            queryPoints[i][j] = dis(gen);
        }
    }
    return queryPoints;
}

// CPU端的主要处理函数
void knnGPU(const std::vector<std::vector<double>>& data,
    const std::vector<std::vector<double>>& queries,
    std::vector<std::vector<std::pair<int, double>>>& results) {

    double* d_data, * d_query, * d_knnDistances;
    int* d_knnIndices;

    int numBlocks = (NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t sharedMemSize = BLOCK_SIZE * sizeof(double);

    // 分配 GPU 内存，确保大小正确
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, NUM_POINTS * DIMENSIONS * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_query, DIMENSIONS * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_knnDistances, numBlocks * K * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_knnIndices, numBlocks * K * sizeof(int)));

    std::vector<double> flatData(NUM_POINTS * DIMENSIONS);
    for (int i = 0; i < NUM_POINTS; ++i) {
        for (int j = 0; j < DIMENSIONS; ++j) {
            flatData[i * DIMENSIONS + j] = data[i][j];
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, flatData.data(),
        NUM_POINTS * DIMENSIONS * sizeof(double),
        cudaMemcpyHostToDevice));

    for (int q = 0; q < queries.size(); ++q) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_query, queries[q].data(),
            DIMENSIONS * sizeof(double),
            cudaMemcpyHostToDevice));

        computeKNN << <numBlocks, BLOCK_SIZE, sharedMemSize >> > (d_data, d_query,
            d_knnDistances, d_knnIndices, NUM_POINTS);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // 使用 Thrust 对结果排序
        thrust::device_vector<double> distances(d_knnDistances, d_knnDistances + numBlocks * K);
        thrust::device_vector<int> indices(d_knnIndices, d_knnIndices + numBlocks * K);

        thrust::sort_by_key(distances.begin(), distances.end(), indices.begin());

        results[q].resize(K);
        for (int i = 0; i < K; ++i) {
            results[q][i] = { indices[i], distances[i] };
        }
    }

    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_query));
    CHECK_CUDA_ERROR(cudaFree(d_knnDistances));
    CHECK_CUDA_ERROR(cudaFree(d_knnIndices));
}

int main() {
    std::string filename = "knn_large_data.txt";
    generateData(filename);

    std::vector<std::vector<double>> data;
    std::cout << "Loading data..." << std::endl;
    loadData(filename, data);

    std::vector<std::vector<double>> queryPoints = generateQueryPoints();
    std::vector<std::vector<std::pair<int, double>>> results(NUM_QUERIES);

    auto start = std::chrono::high_resolution_clock::now();

    knnGPU(data, queryPoints, results);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "GPU Processing time: " << duration.count() << " ms" << std::endl;

    std::cout << "\nFirst query point results:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Index: " << results[0][i].first
            << " Distance: " << results[0][i].second << std::endl;
    }

    return 0;
}