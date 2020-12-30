#include <iostream>
#include <string>
#include <iomanip>
#include <cstdio>
#include <cstdint>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaProfiler.h>

#ifndef COMMON_H
#define COMMON_H

using namespace std;

// Number of times to run each kernel
#define NR_ITERATIONS 10

// Number of times to run each benchmark
#define NR_BENCHMARKS 5

// Helper functions
unsigned roundToPowOf2(unsigned number);

// Function to run a set of kernels
void run(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties);

// Function to run a single kernel
double run_kernel(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties,
    void *kernel,
    float *ptr,
    dim3 gridDim,
    dim3 blockDim);

// Function to report kernel performance
void report(
    string name,
    double milliseconds,
    double gflops = 0,
    double gbytes = 0,
    double gops   = 0);

#endif // end COMMON_H
