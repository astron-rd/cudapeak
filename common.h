#include <iostream>
#include <string>
#include <iomanip>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef COMMON_H
#define COMMON_H

#if defined(HAVE_PMT)
#include <pmt.h>
#endif

using namespace std;

// Number of times to run each kernel
#define NR_ITERATIONS 1

// Number of times to run each benchmark
#define NR_BENCHMARKS 100

// Helper functions
unsigned roundToPowOf2(unsigned number);

// Function to run a set of kernels
void run(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties
#if defined(HAVE_PMT)
    , std::shared_ptr<pmt::PMT> pmt
#endif
    );

typedef struct {
    double runtime; // milliseconds
    double power; // watts
} measurement;

// Function to run a single kernel
measurement run_kernel(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties,
    void *kernel,
    void *ptr,
    dim3 gridDim,
    dim3 blockDim
#if defined(HAVE_PMT)
    , std::shared_ptr<pmt::PMT> pmt
#endif
    );

// Function to report kernel performance
void report(
    string name,
    measurement measurement,
    double gflops = 0,
    double gbytes = 0,
    double gops   = 0);

#endif // end COMMON_H
